import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pywt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
import warnings
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import json

# RBA imports
import core.model as model
import core.helpers as helpers
from core.database import RBAThetaDB
import core.event_extraction as ee

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device configuration
device = torch.device('cpu')
print(f"Using device: {device}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: CONFIGURATION (RESOLUTION-AGNOSTIC)
# ═══════════════════════════════════════════════════════════════════════════

class EnhancedDWTConfig:
    """Resolution-agnostic configuration with ablation controls"""
    
    # ========== ABLATION CONTROLS ==========
    USE_RBA_FEATURES = True
    USE_DWT = True
    USE_SHORT_HORIZON_OPT = True
    USE_FREQUENCY_AWARE_PREDICTION = True
    USE_MAB_FEATURE_SELECTION = True  # NEW! Multi-Armed Bandit
    
    # ========== DATA PATHS ==========
    DATA_PATH = "./input_data/Baltic_Eagle_with_ERA5_Weather.xlsx"
    TIME_COLUMN = "time"
    TARGET_COLUMN = "Power"  # Can be list: ["Power", "Voltage", "Current"]
    
    # ========== AUTO-DETECTION ==========
    AUTO_DETECT_RESOLUTION = True
    MANUAL_RESOLUTION_MINUTES = None
    
    # ========== DWT PARAMETERS ==========
    WAVELET = 'db4'
    DWT_LEVEL = 4
    FREQUENCY_BANDS = ['approximation', 'details_4', 'details_3', 'details_2', 'details_1']
    
    # ========== HAWKES PROCESS SETTINGS ==========
    USE_HAWKES_PROCESS = True
    HAWKES_DIM = 24
    HAWKES_KERNELS = 3
    HAWKES_DROPOUT = 0.45
    
    # ========== MODEL ARCHITECTURE ==========
    SEQUENCE_LENGTH_HOURS = 48
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.45
    
    # ========== PREDICTION HORIZONS (HOURS) ==========
    PREDICTION_HORIZONS_HOURS = [1, 6, 12, 24]
    
    # ========== SHORT-HORIZON OPTIMIZATION ==========
    SHORT_HORIZON_THRESHOLD_HOURS = 1.0
    PERSISTENCE_WEIGHT = 0.7
    
    # ========== TRAINING ==========
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    PATIENCE = 10
    N_SPLITS = 5
    
    # ========== MAB FEATURE SELECTION ==========
    USE_MAB_FEATURE_SELECTION = True
    MAB_ROUNDS = 80  # Number of exploration rounds
    MAB_EPSILON = 0.35  # Exploration rate (35% random)
    MAB_INITIAL_BUDGET = 12  # Features to test per round
    MAB_NONLINEAR_FEATURES = True  # Enable non-linear discovery
    
    # ========== FEATURE SELECTION ==========
    MAX_FEATURES = 75

    # FEATURE QUOTAS 
    MAB_FEATURE_QUOTAS = {
        'rba': 15,          # PRIORITY: RBA event features
        'dwt': 10,          # DWT frequency bands
        'weather': 8,      # Weather patterns
        'power': 5,         # Power/target features
        'temporal': 3,      # Time features
        'nonlinear': 15     # Non-linear relationships
    }

    # ========== STAGED TRAINING ==========
    HAWKES_PRETRAIN_EPOCHS = 5
    USE_STAGED_TRAINING = False
    
    # ========== OUTPUT ==========
    RECONSTRUCT_SAMPLES = 2000
    RESULTS_DIR = "final_experiment_transformer/"
    SAVE_MODELS = True
    MODEL_CHECKPOINT_DIR = "model_checkpoints/"
    PREDICTIONS_DIR = "predictions/"
    WORKFLOW_DIR = "workflow_artifacts/"
    # Model save paths (will be created under RESULTS_DIR)
    def get_model_save_path(self):
        """Get full model checkpoint path"""
        import os
        path = os.path.join(self.RESULTS_DIR, self.MODEL_CHECKPOINT_DIR)
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_predictions_save_path(self):
        """Get predictions save path"""
        import os
        path = os.path.join(self.RESULTS_DIR, self.PREDICTIONS_DIR)
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_workflow_save_path(self):
        """Get workflow artifacts path"""
        import os
        path = os.path.join(self.RESULTS_DIR, self.WORKFLOW_DIR)
        os.makedirs(path, exist_ok=True)
        return path
    
    # ========== RUNTIME ==========
    resolution_minutes = None
    resolution_label = None
    sequence_length = None
    prediction_horizons = None
    short_horizon_threshold = None
    target_columns = None  # NEW! List of target columns
    is_multivariate = False  # NEW! Flag for multivariate
    
    def detect_and_set_resolution(self, df: pd.DataFrame):
        """Auto-detect temporal resolution from data"""
        
        if self.MANUAL_RESOLUTION_MINUTES is not None:
            self.resolution_minutes = self.MANUAL_RESOLUTION_MINUTES
            print(f"Using MANUAL resolution: {self.resolution_minutes} minutes")
        
        elif self.AUTO_DETECT_RESOLUTION and self.TIME_COLUMN in df.columns:
            time_col = pd.to_datetime(df[self.TIME_COLUMN])
            time_diffs = time_col.diff().dropna()
            median_diff = time_diffs.median()
            self.resolution_minutes = median_diff.total_seconds() / 60
            
            print(f"\n{'='*80}")
            print("AUTO-DETECTED TEMPORAL RESOLUTION")
            print(f"{'='*80}")
            print(f"Time differences: Min={time_diffs.min()}, Median={time_diffs.median()}, Max={time_diffs.max()}")
            print(f"✓ Detected resolution: {self.resolution_minutes:.2f} minutes")
            print(f"{'='*80}\n")
        else:
            self.resolution_minutes = 60
            print(f"⚠ Using DEFAULT resolution: {self.resolution_minutes} minutes")
        
        if self.resolution_minutes < 1:
            self.resolution_label = f"{int(self.resolution_minutes * 60)}sec"
        elif self.resolution_minutes < 60:
            self.resolution_label = f"{int(self.resolution_minutes)}min"
        elif self.resolution_minutes == 60:
            self.resolution_label = "1h"
        elif self.resolution_minutes % 60 == 0:
            self.resolution_label = f"{int(self.resolution_minutes / 60)}h"
        else:
            self.resolution_label = f"{self.resolution_minutes:.1f}min"
        
        self._convert_to_timesteps()
    
    def detect_target_columns(self, df: pd.DataFrame):
        """Detect if univariate or multivariate"""
        if isinstance(self.TARGET_COLUMN, list):
            self.target_columns = self.TARGET_COLUMN
            self.is_multivariate = len(self.target_columns) > 1
        else:
            self.target_columns = [self.TARGET_COLUMN]
            self.is_multivariate = False
        
        print(f"\n{'='*80}")
        print("TARGET DETECTION")
        print(f"{'='*80}")
        print(f"Mode: {'MULTIVARIATE' if self.is_multivariate else 'UNIVARIATE'}")
        print(f"Target columns: {self.target_columns}")
        print(f"{'='*80}\n")
    
    def _convert_to_timesteps(self):
        """Convert hour-based parameters to timesteps"""
        
        self.sequence_length = int((self.SEQUENCE_LENGTH_HOURS * 60) / self.resolution_minutes)
        
        self.prediction_horizons = []
        seen_timesteps = set()
        
        for h_hours in self.PREDICTION_HORIZONS_HOURS:
            h_timesteps = int((h_hours * 60) / self.resolution_minutes)
            if h_timesteps == 0:
                h_timesteps = 1
            
            if h_timesteps in seen_timesteps:
                print(f"⚠ Skipping {h_hours}h horizon - conflicts with existing {h_timesteps} timestep horizon")
                continue
            
            seen_timesteps.add(h_timesteps)
            self.prediction_horizons.append(h_timesteps)
        
        self.PREDICTION_HORIZONS_HOURS = [self.timesteps_to_hours(t) for t in self.prediction_horizons]
        
        self.short_horizon_threshold = int(
            (self.SHORT_HORIZON_THRESHOLD_HOURS * 60) / self.resolution_minutes
        )
        
        print(f"PARAMETER CONVERSION (Hours → Timesteps)")
        print(f"{'─'*80}")
        print(f"Resolution: {self.resolution_label} ({self.resolution_minutes:.1f} min/timestep)")
        print(f"Sequence: {self.SEQUENCE_LENGTH_HOURS}h → {self.sequence_length} timesteps")
        print(f"Valid Horizons: {self.PREDICTION_HORIZONS_HOURS} hours → {self.prediction_horizons} timesteps")
        print(f"{'─'*80}\n")
    
    def hours_to_timesteps(self, hours: float) -> int:
        """Convert hours to timesteps"""
        if self.resolution_minutes is None:
            raise ValueError("Resolution not set!")
        return max(1, int((hours * 60) / self.resolution_minutes))
    
    def timesteps_to_hours(self, timesteps: int) -> float:
        """Convert timesteps to hours"""
        if self.resolution_minutes is None:
            raise ValueError("Resolution not set!")
        return (timesteps * self.resolution_minutes) / 60
    
    def get_window_sizes_for_features(self):
        """Get rolling window sizes in timesteps"""
        return [
            self.hours_to_timesteps(6),
            self.hours_to_timesteps(12),
            self.hours_to_timesteps(24)
        ]
    
    def get_lag_sizes_for_features(self):
        """Get lag sizes in timesteps"""
        return [
            self.hours_to_timesteps(1),
            self.hours_to_timesteps(6),
            self.hours_to_timesteps(12),
            self.hours_to_timesteps(24)
        ]
    
    def get_ablation_name(self):
        """Generate name based on settings"""
        name = f"Enhanced_B_DWT_{self.resolution_label}"
        if not self.USE_RBA_FEATURES:
            name += "_RawOnly"
        if not self.USE_DWT:
            name += "_NoDWT"
        if not self.USE_SHORT_HORIZON_OPT:
            name += "_NoPersistence"
        if self.USE_FREQUENCY_AWARE_PREDICTION:
            name += "_FreqAware"
        if self.USE_MAB_FEATURE_SELECTION:
            name += "_MAB"
        if self.is_multivariate:
            name += f"_MV{len(self.target_columns)}"
        return name
    
    def __str__(self):
        res_info = f"{self.resolution_label} ({self.resolution_minutes:.1f} min)" if self.resolution_minutes else "Not detected"
        return f"""
{'='*80}
ENHANCED BASELINE B - CONFIGURATION
{'='*80}
Ablation: {self.get_ablation_name()}
  RBA Features:     {self.USE_RBA_FEATURES}
  DWT:              {self.USE_DWT}
  Short-Horizon:    {self.USE_SHORT_HORIZON_OPT}
  Freq-Aware:       {self.USE_FREQUENCY_AWARE_PREDICTION}

Resolution:         {res_info}
Horizons:           {self.PREDICTION_HORIZONS_HOURS} hours
Sequence:           {self.SEQUENCE_LENGTH_HOURS} hours
{'='*80}
"""

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: DWT
# ═══════════════════════════════════════════════════════════════════════════

class DWTDecomposer:
    """Discrete Wavelet Transform for multi-resolution analysis"""
    
    def __init__(self, config):
        self.config = config
        self.wavelet = config.WAVELET
        self.level = config.DWT_LEVEL
        
    def decompose(self, signal: np.ndarray) -> dict:
        """Decompose signal into frequency bands"""
        
        print(f"\n{'='*80}")
        print("DWT DECOMPOSITION")
        print(f"{'='*80}\n")
        print(f"Signal length: {len(signal)}, Wavelet: {self.wavelet}, Level: {self.level}")
        
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        
        decomposition = {
            'approximation': coeffs[0],
            'original_length': len(signal)
        }
        
        for i in range(1, len(coeffs)):
            decomposition[f'details_{i}'] = coeffs[i]
            print(f"  Detail {i}: {len(coeffs[i])} coefficients")
        
        print(f"✓ Decomposition complete\n{'='*80}\n")
        return decomposition
    
    def reconstruct(self, decomposition: dict) -> np.ndarray:
        """Reconstruct signal from DWT coefficients"""
        coeffs = [decomposition['approximation']]
        for i in range(1, self.level + 1):
            coeffs.append(decomposition[f'details_{i}'])
        
        reconstructed = pywt.waverec(coeffs, self.wavelet)
        return reconstructed[:decomposition['original_length']]
    
    def reconstruct_from_bands(self, decomposition: dict, bands_to_use: list) -> np.ndarray:
        """Reconstruct using only specific bands"""
        coeffs = [np.zeros_like(decomposition['approximation'])]
        for i in range(1, self.level + 1):
            coeffs.append(np.zeros_like(decomposition[f'details_{i}']))
        
        if 'approximation' in bands_to_use:
            coeffs[0] = decomposition['approximation']
        
        for i in range(1, self.level + 1):
            if f'details_{i}' in bands_to_use:
                coeffs[i] = decomposition[f'details_{i}']
        
        reconstructed = pywt.waverec(coeffs, self.wavelet)
        return reconstructed[:decomposition['original_length']]

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: MULTI-RESOLUTION EVENT EXTRACTOR (UNCHANGED - Already has per-band)
# ═══════════════════════════════════════════════════════════════════════════

class MultiResolutionEventExtractor:
    """Extract events from frequency bands with dynamic thresholds and DATA FUSION"""
    
    def __init__(self, config):
        self.config = config
        
    def _calculate_dynamic_threshold_factor(self, signal: np.ndarray, 
                                           band_name: str, 
                                           band_level: int = None) -> float:
        """Calculate data-driven threshold factor"""
        
        signal_std = np.std(signal)
        signal_var = np.var(signal)
        signal_mean_abs = np.mean(np.abs(signal))
        
        # Robust noise estimation using MAD
        mad = np.median(np.abs(signal - np.median(signal)))
        noise_estimate = 1.4826 * mad
        snr_estimate = signal_std / noise_estimate if noise_estimate > 0 else 1.0
        
        # Coefficient of variation
        cv = signal_std / signal_mean_abs if signal_mean_abs > 0 else 1.0
        
        if band_name == 'approximation':
            # Trend: higher threshold (less sensitive)
            base_factor = 1.5
            stability_adjustment = 1.0 + (0.5 / (cv + 0.1))
            threshold_factor = base_factor * stability_adjustment
        else:
            # Details: frequency-dependent
            if band_level is None:
                band_level = int(band_name.split('_')[1])
            
            base_factor = 0.4 + (band_level * 0.15)
            
            energy_percentile = np.percentile(np.abs(signal), 90)
            energy_factor = np.clip(signal_std / energy_percentile, 0.5, 2.0) if energy_percentile > 0 else 1.0
            
            snr_factor = np.clip(2.0 / (snr_estimate + 1.0), 0.7, 1.5)
            
            threshold_factor = base_factor * energy_factor * snr_factor
        
        return np.clip(threshold_factor, 0.3, 2.5)
    
    def extract_events_per_band(self, decomposition: dict, nominal: float, 
                                param_config: dict) -> dict:
        """Extract RBA events with dynamic thresholds + DATA FUSION"""
        
        print(f"\n{'='*80}")
        print("MULTI-RESOLUTION EVENT EXTRACTION (DYNAMIC THRESHOLDS + FUSION)")
        print(f"{'='*80}\n")
        
        import core.model as model
        import core.event_extraction as ee
        
        all_band_events = {}
        threshold_report = []
        original_length = decomposition['original_length']
        
        # Approximation
        print(f"Extracting from approximation...")
        approx_signal = decomposition['approximation']
        approx_upsampled = np.interp(
            np.linspace(0, len(approx_signal), original_length),
            np.arange(len(approx_signal)),
            approx_signal
        )
        
        threshold_factor = self._calculate_dynamic_threshold_factor(approx_upsampled, 'approximation')
        print(f"  Dynamic threshold: {threshold_factor:.3f}")
        
        threshold_report.append({
            'band': 'approximation',
            'threshold_factor': threshold_factor,
            'signal_std': np.std(approx_upsampled)
        })
        
        approx_events = self._extract_from_signal(approx_upsampled, nominal, param_config, threshold_factor)
        if not approx_events.empty:
            approx_events['frequency_band'] = 'approximation'
            all_band_events['approximation'] = approx_events
            print(f"  ✓ {len(approx_events)} events")
        
        # Details
        for i in range(1, self.config.DWT_LEVEL + 1):
            print(f"\nExtracting from details_{i}...")
            detail_signal = decomposition[f'details_{i}']
            detail_upsampled = np.interp(
                np.linspace(0, len(detail_signal), original_length),
                np.arange(len(detail_signal)),
                detail_signal
            )
            
            threshold_factor = self._calculate_dynamic_threshold_factor(detail_upsampled, f'details_{i}', i)
            print(f"  Dynamic threshold: {threshold_factor:.3f}")
            
            threshold_report.append({
                'band': f'details_{i}',
                'threshold_factor': threshold_factor,
                'signal_std': np.std(detail_upsampled)
            })
            
            detail_events = self._extract_from_signal(detail_upsampled, nominal, param_config, threshold_factor)
            if not detail_events.empty:
                detail_events['frequency_band'] = f'details_{i}'
                all_band_events[f'details_{i}'] = detail_events
                print(f"  ✓ {len(detail_events)} events")
        
        # Summary
        print(f"\n{'─'*80}")
        print(f"{'Band':<15} {'Threshold':<12} {'Std Dev':<12} {'Events':<10}")
        print(f"{'─'*80}")
        raw_event_count = 0
        for item in threshold_report:
            band_name = item['band']
            event_count = len(all_band_events[band_name]) if band_name in all_band_events else 0
            raw_event_count += event_count
            print(f"{item['band']:<15} {item['threshold_factor']:<12.3f} {item['signal_std']:<12.4f} {event_count:<10}")
        print(f"{'─'*80}")
        
        # DATA FUSION: Combine overlapping events from different bands
        print(f"\n DATA FUSION: Combining {raw_event_count} raw detections...")
        combined_events = pd.concat(all_band_events.values(), ignore_index=True) if all_band_events else pd.DataFrame()
        
        if not combined_events.empty:
            fused_events = self._fuse_multi_resolution_events(combined_events)
            print(f"✓ Fused into {len(fused_events)} high-confidence events")
        else:
            fused_events = pd.DataFrame()
        
        print(f"\n{'='*80}\n")
        
        return {'per_band': all_band_events, 'combined': fused_events, 'threshold_report': threshold_report}
    
    def _fuse_multi_resolution_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        DATA FUSION STRATEGY: Weighted Fusion with Confidence Scoring
        """
    
        if events_df.empty:
            return events_df
        
        # Frequency weights (physics-informed: ramp bands get highest weight)
        frequency_weights = {
            'approximation': 0.5,  # Slow trends
            'details_1': 0.5,      # High-freq noise (too fast)
            'details_2': 0.8,      # Short ramps
            'details_3': 1.0,      # MAIN RAMP BAND ← Most important!
            'details_4': 0.85       # Medium ramps / daily patterns
        }
        
        print(f"  Frequency weights: {frequency_weights}")
        
        # Sort by start time
        events_df = events_df.sort_values('t1').reset_index(drop=True)
        
        fused_events = []
        multi_band_threshold = 0.1
        single_band_weight_threshold = 0.8
        
        i = 0
        
        while i < len(events_df):
            current_event = events_df.iloc[i]
            current_start = current_event['t1']
            current_end = current_event['t2']
            
            # Find all overlapping events (must have temporal overlap AND be from different bands)
            overlapping = []
            overlapping_bands = set()
            
            # Add current event
            overlapping.append(current_event)
            overlapping_bands.add(current_event['frequency_band'])
            
            j = i + 1
            
            # Look for overlapping events from OTHER bands
            while j < len(events_df) and events_df.iloc[j]['t1'] <= current_end:
                candidate = events_df.iloc[j]
                candidate_band = candidate['frequency_band']
                
                # Only add if from a DIFFERENT band (avoid double-counting)
                if candidate_band not in overlapping_bands:
                    overlapping.append(candidate)
                    overlapping_bands.add(candidate_band)
                    current_end = max(current_end, candidate['t2'])
                
                j += 1
            
            # FUSE overlapping events from different bands
            if len(overlapping) > 1:
                # Weighted magnitude combination
                weighted_magnitudes = []
                total_weight = 0
                
                for event in overlapping:
                    band = event['frequency_band']
                    weight = frequency_weights.get(band, 0.5)
                    
                    # Get magnitude
                    if event['event_type'] == 'significant' and pd.notna(event.get('∆w_m')):
                        mag = abs(event['∆w_m'])
                    elif event['event_type'] == 'stationary' and pd.notna(event.get('σ_s')):
                        mag = event['σ_s']
                    else:
                        mag = 0
                    
                    weighted_magnitudes.append(mag * weight)
                    total_weight += weight
                
                # Weighted average magnitude
                fused_magnitude = sum(weighted_magnitudes) / total_weight if total_weight > 0 else 0
                
                # ✅ Weight-aware confidence
                confidence = sum(frequency_weights.get(e['frequency_band'], 0.5) for e in overlapping) / 5.0
                confidence = min(confidence, 1.0)
                
                # Find dominant band (highest weighted contribution)
                dominant_event = max(overlapping, key=lambda e: 
                    (abs(e.get('∆w_m', 0)) if pd.notna(e.get('∆w_m')) else e.get('σ_s', 0)) * 
                    frequency_weights.get(e['frequency_band'], 0.5)
                )
                
                # Create fused event
                fused_event = dominant_event.copy()
                fused_event['∆w_m'] = fused_magnitude
                fused_event['t1'] = min(e['t1'] for e in overlapping)
                fused_event['t2'] = max(e['t2'] for e in overlapping)
                fused_event['fusion_confidence'] = confidence
                fused_event['num_bands'] = len(overlapping_bands)
                
                # ✅ Keep if confidence ≥ multi_band_threshold
                if confidence >= multi_band_threshold:
                    fused_events.append(fused_event)
                
            else:
                # Single-band events: only keep from important bands
                band = current_event['frequency_band']
                weight = frequency_weights.get(band, 0)
                
                # ✅ Keep if weight > single_band_weight_threshold
                if weight >= single_band_weight_threshold:
                    current_event['fusion_confidence'] = weight * 0.2
                    current_event['num_bands'] = 1
                    fused_events.append(current_event)
            
            # Skip ahead to avoid reprocessing
            i = j if j > i + 1 else i + 1
        
        result = pd.DataFrame(fused_events)
        
        # 🔍 DEBUGGING: Count multi-band vs single-band
        multi_band_count = sum(1 for e in fused_events if e.get('num_bands', 1) > 1)
        single_band_count = sum(1 for e in fused_events if e.get('num_bands', 1) == 1)
        
        # Count single-band events by band
        single_band_breakdown = {}
        for e in fused_events:
            if e.get('num_bands', 1) == 1:
                band = e.get('frequency_band', 'unknown')
                single_band_breakdown[band] = single_band_breakdown.get(band, 0) + 1
        
        # Summary statistics
        print(f"  Input: {len(events_df)} raw detections across bands")
        print(f"  Fused: {len(result)} events")
        print(f"    - Multi-band (confidence ≥ {multi_band_threshold}): {multi_band_count}")
        print(f"    - Single-band (weight > {single_band_weight_threshold}): {single_band_count}")
        
        if single_band_breakdown:
            print(f"    - Single-band breakdown:")
            for band in sorted(single_band_breakdown.keys()):
                print(f"        {band}: {single_band_breakdown[band]}")
        
        if not result.empty:
            print(f"  Avg confidence: {result['fusion_confidence'].mean():.2f}")
            print(f"  Avg bands per event: {result['num_bands'].mean():.1f}")
            print(f"  Confidence distribution: min={result['fusion_confidence'].min():.2f}, max={result['fusion_confidence'].max():.2f}")
        
        return result
    
    def _extract_from_signal(self, signal: np.ndarray, nominal: float, param_config: dict, threshold_factor: float) -> pd.DataFrame:
        """Extract RBA events from signal"""
        import core.model as model
        import core.event_extraction as ee
        
        normalized = signal / nominal
        adaptive_threshold = model.calculate_adaptive_threshold(normalized)
        
        sig_threshold = adaptive_threshold * param_config.get("trad_sig_event_factor", 0.00008) * threshold_factor
        stat_threshold = adaptive_threshold * param_config.get("trad_stat_event_factor", 0.000024) * threshold_factor
        
        sig_events = ee.significant_events(
            data=normalized, threshold=sig_threshold,
            min_duration=param_config.get("trad_min_duration", 3),
            min_slope=param_config.get("trad_min_slope", 0.03),
            window_minutes=param_config.get("trad_window", 30),
            freq_secs=param_config.get("trad_freq_secs", 800),
        )
        if not sig_events.empty:
            sig_events['event_type'] = 'significant'
        
        stat_events = ee.stationary_events(
            data=normalized, threshold=stat_threshold,
            min_duration=param_config.get("trad_min_duration", 3),
            min_stationary_length=param_config.get("trad_min_stationary_length", 4),
            window_minutes=param_config.get("trad_window", 30),
            freq_secs=param_config.get("trad_freq_secs", 800),
        )
        if not stat_events.empty:
            stat_events['event_type'] = 'stationary'
        
        all_events = []
        if not sig_events.empty:
            all_events.append(sig_events)
        if not stat_events.empty:
            all_events.append(stat_events)
        
        return pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()

# ════════════════════════════════════════════════════════════
# SECTION 3B: MULTI-ARMED BANDIT FEATURE SELECTOR (FIXED!)
# ════════════════════════════════════════════════════════════
class StratifiedMABFeatureSelector:
    """
    MAB with GUARANTEED feature group representation + Temporal downsampling
    
    Strategy:
    1. Divide budget among feature groups (RBA, DWT, Weather, Non-linear)
    2. Each group gets minimum quota
    3. Remaining slots go to best performers
    4. Temporal-aware downsampling preserves time series structure
    """
    
    def __init__(self, config):
        self.config = config
        self.n_rounds = config.MAB_ROUNDS
        self.epsilon = config.MAB_EPSILON
        self.budget = config.MAB_INITIAL_BUDGET
        
        # Feature group quotas (guaranteed minimums)
        self.feature_quotas = {
            'rba': 15,
            'dwt': 10,      # Reduce from 12
            'weather': 8,    # Force weather (was 10)
            'power': 5,      # Reduce from 8
            'temporal': 3,   # Reduce from 5
            'nonlinear': 15  # Reduce from 25
        }
    
    def _temporal_downsample(self, X: np.ndarray, y: np.ndarray, 
                        target_size: int, max_horizon: int) -> tuple:
        """
        Downsample while preserving temporal structure
        
        Strategy:
        1. Divide timeline into chunks
        2. Sample chunks (not individual points)
        3. Ensure each chunk has enough context for max_horizon prediction
        """
        
        n_samples = len(X)
        
        if n_samples <= target_size:
            return X, y
        
        print(f"\n  📊 Temporal-aware downsampling:")
        print(f"     Original: {n_samples:,} samples")
        print(f"     Target: {target_size:,} samples")
        print(f"     Max horizon: {max_horizon}h")
        
        # Define chunk size (must be > max_horizon for temporal context)
        # Use 3x the max horizon to ensure sufficient context
        chunk_size = int(max(max_horizon * 3, 72))  # ✅ FIXED: Cast to int!
        
        print(f"     Chunk size: {chunk_size}h (3x horizon)")
        
        # Calculate how many chunks we need
        n_chunks_total = n_samples // chunk_size
        n_chunks_needed = max(1, target_size // chunk_size)  # ✅ Already int from //
        
        print(f"     Chunks: {n_chunks_needed}/{n_chunks_total}")  # Debug info
        
        if n_chunks_needed >= n_chunks_total:
            # Need almost all data, use stride sampling
            stride = max(1, n_samples // target_size)
            indices = np.arange(0, n_samples, stride)[:target_size]
            print(f"     Method: Stride sampling (every {stride} samples)")
        else:
            # Randomly select chunks (preserves temporal structure within chunks)
            chunk_starts = np.arange(0, n_samples - chunk_size, chunk_size)
            
            # ✅ FIXED: Ensure n_chunks_needed is int
            n_chunks_needed = int(n_chunks_needed)
            
            # Randomly select chunk start positions
            selected_chunk_starts = np.random.choice(
                chunk_starts, 
                size=n_chunks_needed,  # Now guaranteed to be int
                replace=False
            )
            selected_chunk_starts.sort()  # Keep temporal order
            
            # Build indices from selected chunks
            indices = []
            for start in selected_chunk_starts:
                chunk_indices = list(range(start, min(start + chunk_size, n_samples)))
                indices.extend(chunk_indices)
            
            indices = np.array(indices)[:target_size]
            
            print(f"     Method: Chunk sampling")
            print(f"       Chunks selected: {n_chunks_needed}/{n_chunks_total}")
        
        # Sample data
        X_sampled = X[indices]
        y_sampled = y[indices]
        
        # Verify class balance is maintained
        original_pos_rate = y.mean()
        sampled_pos_rate = y_sampled.mean()
        
        print(f"     Result: {len(indices):,} samples")
        print(f"     Positive rate: {original_pos_rate*100:.2f}% → {sampled_pos_rate*100:.2f}%")
        
        # Check if balance is severely disrupted
        if abs(original_pos_rate - sampled_pos_rate) > 0.05:
            print(f"     ⚠️ Class imbalance detected (>{5}% drift)")
            print(f"     Applying stratified adjustment...")
            X_sampled, y_sampled = self._stratified_chunk_sample(
                X, y, indices, original_pos_rate
            )
            print(f"     ✓ Adjusted positive rate: {y_sampled.mean()*100:.2f}%")
        
        return X_sampled, y_sampled
    
    def _stratified_chunk_sample(self, X: np.ndarray, y: np.ndarray, 
                                 indices: np.ndarray, target_pos_rate: float) -> tuple:
        """
        Adjust chunk sampling to maintain class balance
        
        Strategy:
        1. Separate indices by class
        2. Resample each class to match target distribution
        3. Sort to maintain temporal order
        """
        
        # Split indices by class
        pos_indices = indices[y[indices] == 1]
        neg_indices = indices[y[indices] == 0]
        
        # Calculate target counts
        n_total = len(indices)
        n_pos_target = int(n_total * target_pos_rate)
        n_neg_target = n_total - n_pos_target
        
        # Sample to match target rates
        if len(pos_indices) > n_pos_target:
            pos_indices = np.random.choice(pos_indices, n_pos_target, replace=False)
        elif len(pos_indices) < n_pos_target:
            # Need more positive samples - sample with replacement
            pos_indices = np.random.choice(pos_indices, n_pos_target, replace=True)
        
        if len(neg_indices) > n_neg_target:
            neg_indices = np.random.choice(neg_indices, n_neg_target, replace=False)
        elif len(neg_indices) < n_neg_target:
            # Need more negative samples - sample with replacement
            neg_indices = np.random.choice(neg_indices, n_neg_target, replace=True)
        
        # Combine and sort to maintain temporal order
        final_indices = np.concatenate([pos_indices, neg_indices])
        final_indices.sort()
        
        return X[final_indices], y[final_indices]
    
    def _categorize_features(self, feature_names: list) -> dict:
        """Categorize features into groups"""
        
        categories = {
            'rba': [],
            'dwt': [],
            'weather': [],
            'power': [],
            'temporal': [],
            'nonlinear': [],
            'other': []
        }
        
        for i, name in enumerate(feature_names):
            name_lower = name.lower()
            
            # Check if non-linear first
            if any(marker in name for marker in ['^2', '^3', '×', 'sqrt(', 'log(', ' / ']):
                categories['nonlinear'].append(i)
            
            # Then check domain categories
            elif any(kw in name_lower for kw in ['event', 'rba', 'magnitude', 'duration', 'timing', 'stationary', 'significant']):
                categories['rba'].append(i)
            
            elif any(kw in name_lower for kw in ['dwt', 'details_', 'approximation', 'freq_ratio']):
                categories['dwt'].append(i)
            
            elif any(kw in name_lower for kw in ['pressure', 'wind', 'temp', 'humidity', 'cloud', 'rain', 'moisture']):
                categories['weather'].append(i)
            
            elif any(kw in name_lower for kw in ['power', 'mean_', 'std_', 'min_', 'max_', 'range_', 'lag_', 'diff_']):
                categories['power'].append(i)
            
            elif any(kw in name_lower for kw in ['hour', 'day', 'month', 'sin', 'cos']):
                categories['temporal'].append(i)
            
            else:
                categories['other'].append(i)
        
        return categories
    
    def _select_best_horizon_for_mab(self, labels: pd.DataFrame, horizons: list) -> tuple:
        """Choose most balanced horizon"""
        
        print(f"\n{'='*80}")
        print("AUTO-SELECTING HORIZON FOR MAB")
        print(f"{'='*80}\n")
        
        horizon_stats = []
        
        for h in horizons:
            col = f'event_occurs_{h}h'
            if col in labels.columns:
                pos_rate = labels[col].mean()
                balance_score = 1.0 - abs(pos_rate - 0.5)
                
                horizon_stats.append({
                    'horizon': h,
                    'pos_rate': pos_rate,
                    'balance_score': balance_score,
                    'column': col
                })
                
                indicator = "⚠️ IMBALANCED" if pos_rate < 0.1 or pos_rate > 0.9 else "✓ Balanced" if 0.4 < pos_rate < 0.6 else "○ OK"
                print(f"  {h:4.1f}h: {pos_rate*100:5.2f}% positive | Balance: {balance_score:.3f} | {indicator}")
        
        best = max(horizon_stats, key=lambda x: x['balance_score'])
        
        print(f"\n  🎯 SELECTED: {best['horizon']}h horizon")
        print(f"     Positive rate: {best['pos_rate']*100:.2f}%")
        print(f"     Reason: Most balanced (closest to 50%)")
        print(f"\n{'='*80}\n")
        
        return best['column'], best['horizon'], best['pos_rate']
    
    def _create_nonlinear_transformations(self, X: np.ndarray, feature_names: list, 
                                          max_new_features: int = 500) -> tuple:
        """Create non-linear features"""
        
        print(f"\n{'='*80}")
        print("DISCOVERING NON-LINEAR RELATIONSHIPS")
        print(f"{'='*80}\n")
        
        n_features = X.shape[1]
        print(f"Input features: {n_features}")
        
        new_features = []
        new_names = []
        
        # Polynomials
        print(f"\n  1. Polynomial features...")
        poly_count = 0
        for i in range(n_features):
            squared = X[:, i] ** 2
            if not np.isnan(squared).any() and not np.isinf(squared).any():
                new_features.append(squared)
                new_names.append(f"{feature_names[i]}^2")
                poly_count += 1
            
            if poly_count < 100:
                cubed = X[:, i] ** 3
                if not np.isnan(cubed).any() and not np.isinf(cubed).any():
                    new_features.append(cubed)
                    new_names.append(f"{feature_names[i]}^3")
                    poly_count += 1
        print(f"    Created {poly_count} features")
        
        # Interactions
        print(f"\n  2. Interaction features...")
        interaction_count = 0
        
        weather_keywords = ['pressure', 'wind', 'temp', 'humidity', 'cloud', 'rain', 'moisture']
        weather_indices = [i for i, name in enumerate(feature_names) 
                          if any(kw in name.lower() for kw in weather_keywords)]
        
        power_keywords = ['power', 'dwt', 'event', 'rba', 'magnitude', 'duration']
        power_indices = [i for i, name in enumerate(feature_names) 
                        if any(kw in name.lower() for kw in power_keywords)]
        
        for i in weather_indices[:20]:
            for j in weather_indices[i+1:20]:
                interaction = X[:, i] * X[:, j]
                if not np.isnan(interaction).any() and not np.isinf(interaction).any():
                    new_features.append(interaction)
                    new_names.append(f"{feature_names[i]} × {feature_names[j]}")
                    interaction_count += 1
                    if interaction_count >= max_new_features // 3:
                        break
            if interaction_count >= max_new_features // 3:
                break
        
        for i in weather_indices[:15]:
            for j in power_indices[:15]:
                interaction = X[:, i] * X[:, j]
                if not np.isnan(interaction).any() and not np.isinf(interaction).any():
                    new_features.append(interaction)
                    new_names.append(f"{feature_names[i]} × {feature_names[j]}")
                    interaction_count += 1
                    if interaction_count >= (max_new_features * 2) // 3:
                        break
            if interaction_count >= (max_new_features * 2) // 3:
                break
        
        print(f"    Created {interaction_count} features")
        
        # Roots
        print(f"\n  3. Root features...")
        root_count = 0
        for i in range(min(50, n_features)):
            sqrt_feature = np.sqrt(np.abs(X[:, i]) + 1e-8)
            if not np.isnan(sqrt_feature).any() and not np.isinf(sqrt_feature).any():
                new_features.append(sqrt_feature)
                new_names.append(f"sqrt({feature_names[i]})")
                root_count += 1
        print(f"    Created {root_count} features")
        
        # Logs
        print(f"\n  4. Log features...")
        log_count = 0
        for i in range(min(50, n_features)):
            log_feature = np.log1p(np.abs(X[:, i]))
            if not np.isnan(log_feature).any() and not np.isinf(log_feature).any():
                new_features.append(log_feature)
                new_names.append(f"log({feature_names[i]})")
                log_count += 1
        print(f"    Created {log_count} features")
        
        # Ratios
        print(f"\n  5. Ratio features...")
        ratio_count = 0
        for i in weather_indices[:10]:
            for j in weather_indices[i+1:10]:
                ratio = X[:, i] / (X[:, j] + 1e-8)
                if not np.isnan(ratio).any() and not np.isinf(ratio).any():
                    new_features.append(ratio)
                    new_names.append(f"{feature_names[i]} / {feature_names[j]}")
                    ratio_count += 1
                    if ratio_count >= 50:
                        break
            if ratio_count >= 50:
                break
        print(f"    Created {ratio_count} features")
        
        if new_features:
            X_new = np.column_stack(new_features)
            X_combined = np.hstack([X, X_new])
            combined_names = feature_names + new_names
            
            print(f"\n  📊 Summary:")
            print(f"     New features: {len(new_features)}")
            print(f"     Total: {X_combined.shape[1]}")
            print(f"     Memory: ~{X_combined.nbytes / (1024**2):.2f} MB")
        else:
            X_combined = X
            combined_names = feature_names
        
        print(f"\n{'='*80}\n")
        
        return X_combined, combined_names
    
    def _evaluate_feature_subset(self, X_subset: np.ndarray, y: np.ndarray) -> float:
        """Evaluate with RandomForest"""
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score, roc_auc_score
        
        try:
            clf = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            f1_scores = []
            auc_scores = []
            
            for train_idx, val_idx in skf.split(X_subset, y):
                X_train, X_val = X_subset[train_idx], X_subset[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)
                y_prob = clf.predict_proba(X_val)[:, 1]
                
                f1 = f1_score(y_val, y_pred, zero_division=0)
                try:
                    auc = roc_auc_score(y_val, y_prob)
                except:
                    auc = 0.5
                
                f1_scores.append(f1)
                auc_scores.append(auc)
            
            return 0.6 * np.mean(f1_scores) + 0.4 * (np.mean(auc_scores) - 0.5) * 2
            
        except:
            return 0.0
    
    def select_features(self, X: np.ndarray, labels: pd.DataFrame, 
                       feature_names: list) -> tuple:
        """
        🎯 STRATIFIED Thompson Sampling with temporal-aware downsampling
        """
        
        print(f"\n{'='*80}")
        print("🔍 STRATIFIED MAB: GUARANTEED RBA REPRESENTATION")
        print(f"{'='*80}\n")
        
        # STEP 1: Auto-select horizon
        target_col, selected_horizon, pos_rate = self._select_best_horizon_for_mab(
            labels, self.config.PREDICTION_HORIZONS_HOURS
        )
        y = labels[target_col].values
        
        # STEP 2: Create non-linear features
        X_expanded, expanded_names = self._create_nonlinear_transformations(
            X, feature_names, max_new_features=500
        )
        
        # STEP 3: Categorize features
        print(f"📂 CATEGORIZING FEATURES...")
        categories = self._categorize_features(expanded_names)
        
        print(f"\n  Feature inventory:")
        for cat, indices in categories.items():
            if cat != 'other' and indices:
                print(f"    {cat.upper():15s}: {len(indices):4d} features | "
                      f"Quota: {self.feature_quotas.get(cat, 0):2d}")
        
        print(f"\n{'='*80}\n")
        
        n_features = X_expanded.shape[1]
        n_samples = X_expanded.shape[0]
        
        # STEP 4: TEMPORAL-AWARE DOWNSAMPLING (FIXED!)
        # Calculate optimal sample size based on dataset and horizon
        min_samples = 10000
        max_samples = 30000  # Increased from 8000!
        
        # Scale based on dataset size (5-10% of total)
        scaled_size = int(n_samples * 0.075)
        
        # Scale based on horizon (longer horizons need more data)
        max_horizon = max(self.config.PREDICTION_HORIZONS_HOURS)
        horizon_based_size = max_horizon * 500
        
        # Take the maximum but cap at max_samples
        target_size = min(max_samples, max(min_samples, scaled_size, horizon_based_size))
        
        print(f"MAB Evaluation Configuration:")
        print(f"  Total samples: {n_samples:,}")
        print(f"  Max horizon: {max_horizon}h")
        print(f"  Target sample size: {target_size:,} ({target_size/n_samples*100:.1f}%)")
        
        if n_samples > target_size:
            X_sampled, y_sampled = self._temporal_downsample(
                X_expanded, y, target_size, max_horizon
            )
        else:
            X_sampled = X_expanded
            y_sampled = y
            print(f"\n  ✓ Using all {n_samples:,} samples (no downsampling needed)")
        
        print(f"\n")
        
        # STEP 5: Stratified Thompson Sampling
        alpha = np.ones(n_features)
        beta = np.ones(n_features)
        feature_scores = np.zeros(n_features)
        feature_counts = np.zeros(n_features)
        
        print(f"🎰 Running STRATIFIED Thompson Sampling ({self.n_rounds} rounds)...\n")
        
        best_score = 0.0
        
        for round_idx in range(self.n_rounds):
            # STRATIFIED SAMPLING: Pick features from each group
            selected_indices = []
            
            for cat, indices in categories.items():
                if not indices or cat == 'other':
                    continue
                
                quota = self.feature_quotas.get(cat, 0)
                if quota == 0:
                    continue
                
                n_pick = min(quota // 2, len(indices))
                if n_pick > 0:
                    cat_alpha = alpha[indices]
                    cat_beta = beta[indices]
                    cat_sampled = np.random.beta(cat_alpha, cat_beta)
                    
                    if np.random.random() < self.epsilon:
                        cat_selected = np.random.choice(indices, size=n_pick, replace=False)
                    else:
                        cat_top = np.argsort(cat_sampled)[-n_pick:]
                        cat_selected = [indices[i] for i in cat_top]
                    
                    selected_indices.extend(cat_selected)
            
            if len(selected_indices) < self.budget:
                remaining = self.budget - len(selected_indices)
                all_sampled = np.random.beta(alpha, beta)
                available = [i for i in range(n_features) if i not in selected_indices]
                top_remaining = np.argsort(all_sampled[available])[-remaining:]
                selected_indices.extend([available[i] for i in top_remaining])
            
            selected_indices = selected_indices[:self.budget]
            
            X_subset = X_sampled[:, selected_indices]
            score = self._evaluate_feature_subset(X_subset, y_sampled)
            
            for idx in selected_indices:
                feature_counts[idx] += 1
                feature_scores[idx] += score
                
                if score > 0.45:
                    alpha[idx] += 3
                elif score > 0.35:
                    alpha[idx] += 2
                elif score > 0.25:
                    alpha[idx] += 1
                else:
                    beta[idx] += 1
            
            if score > best_score:
                best_score = score
            
            if (round_idx + 1) % 5 == 0:
                avg = feature_scores[feature_counts > 0].sum() / (feature_counts[feature_counts > 0].sum() + 1e-8)
                print(f"  Round {round_idx + 1:2d}/{self.n_rounds} | "
                      f"Current: {score:.4f} | Best: {best_score:.4f} | Avg: {avg:.4f}")
        
        # STEP 6: Final STRATIFIED selection
        print(f"\n{'='*80}")
        print("📊 STRATIFIED SELECTION")
        print(f"{'='*80}\n")
        
        avg_scores = np.divide(feature_scores, feature_counts, 
                               out=np.zeros_like(feature_scores), 
                               where=feature_counts > 0)
        
        selected_indices = []
        
        # Select from each category based on quota
        for cat, indices in categories.items():
            if not indices or cat == 'other':
                continue
            
            quota = self.feature_quotas.get(cat, 0)
            if quota == 0:
                continue
            
            cat_scores = avg_scores[indices]
            cat_top_local = np.argsort(cat_scores)[-quota:]
            cat_selected = [indices[i] for i in cat_top_local if cat_scores[i] > 0]
            
            selected_indices.extend(cat_selected)
            
            print(f"  {cat.upper():15s}: Selected {len(cat_selected):2d}/{quota:2d} (quota)")
        
        # Fill remaining slots with best overall
        remaining_slots = self.config.MAX_FEATURES - len(selected_indices)
        if remaining_slots > 0:
            available = [i for i in range(n_features) if i not in selected_indices and avg_scores[i] > 0]
            top_remaining = sorted(available, key=lambda i: avg_scores[i], reverse=True)[:remaining_slots]
            selected_indices.extend(top_remaining)
            print(f"  {'BEST REMAINING':15s}: Selected {len(top_remaining):2d} (fill)")
        
        selected_features = [expanded_names[i] for i in selected_indices]
        
        # Analysis
        original_count = len(feature_names)
        selected_original = sum(1 for name in selected_features if name in feature_names)
        selected_nonlinear = len(selected_features) - selected_original
        
        print(f"\n{'='*80}")
        print("✅ FINAL RESULTS")
        print(f"{'='*80}")
        print(f"  Total selected: {len(selected_features)}")
        print(f"  Original: {selected_original}")
        print(f"  Non-linear: {selected_nonlinear}")
        print(f"  Best score: {best_score:.4f}")
        print(f"  Avg score: {avg_scores[selected_indices].mean():.4f}")
        
        # Category breakdown
        print(f"\n  📂 Category breakdown:")
        for cat in ['rba', 'dwt', 'weather', 'power', 'temporal', 'nonlinear']:
            cat_count = sum(1 for idx in selected_indices if idx in categories.get(cat, []))
            if cat_count > 0:
                print(f"     {cat.upper():15s}: {cat_count:2d}")
        
        print(f"\n  🏅 Top 10 features:")
        top_10_indices = sorted(selected_indices, key=lambda i: avg_scores[i], reverse=True)[:10]
        for rank, idx in enumerate(top_10_indices, 1):
            name = expanded_names[idx]
            is_nonlinear = "🔥" if any(m in name for m in ['^2', '^3', '×', 'sqrt', 'log', '/']) else "📁"
            
            cat_label = "?"
            for cat, cat_indices in categories.items():
                if idx in cat_indices:
                    cat_label = cat[:3].upper()
                    break
            
            print(f"    {rank:2d}. {is_nonlinear} [{cat_label}] {name:<50s} | {avg_scores[idx]:.4f}")
        
        print(f"\n{'='*80}\n")
        
        return X_expanded[:, selected_indices], selected_features

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: SHORT-HORIZON OPTIMIZER (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════

class ShortHorizonOptimizer:
    """Short-term prediction with persistence (from paper)"""
    
    def __init__(self, config):
        self.config = config
        
    def create_persistence_features(self, features: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create persistence features for a specific target"""
        
        if target_col not in features.columns:
            return features
        
        for lag in [1, 2, 3, 4]:
            features[f'{target_col}_persistence_{lag}'] = features[target_col].shift(lag)
        
        features[f'{target_col}_velocity_1t'] = features[target_col].diff(1)
        features[f'{target_col}_velocity_2t'] = features[target_col].diff(2)
        features[f'{target_col}_acceleration'] = features[f'{target_col}_velocity_1t'].diff(1)
        
        for window in [2, 4]:
            features[f'{target_col}_recent_mean_{window}t'] = features[target_col].rolling(window).mean()
            features[f'{target_col}_recent_std_{window}t'] = features[target_col].rolling(window).std()
        
        features[f'{target_col}_persistence_confidence'] = 1.0 / (features[f'{target_col}_recent_std_2t'] + 0.01)
        
        return features

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: FEATURE ENGINEER (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════

class EnhancedFeatureEngineer:
    """Feature engineering with ablation support"""
    
    def __init__(self, config):
        self.config = config
        
    def create_features(self, df: pd.DataFrame, events_dict: dict, dwt_decomposition_dict: dict = None) -> pd.DataFrame:
        """
        Create all features for single or multiple targets
        
        events_dict: {target_col: events_df}
        dwt_decomposition_dict: {target_col: decomposition}
        """
        
        print(f"\n{'='*80}")
        print(f"FEATURE ENGINEERING - {self.config.get_ablation_name()}")
        print(f"{'='*80}\n")
        
        features = self._create_raw_features(df)
        print(f"After raw features: {features.shape}")
    
        if self._has_weather_columns(df):
            features = self._create_weather_interactions(features)
            print(f"After weather interactions: {features.shape}")
        
        # DWT features for each target
        if self.config.USE_DWT and dwt_decomposition_dict:
            for target_col in self.config.target_columns:
                if target_col in dwt_decomposition_dict:
                    features = self._create_dwt_features(features, dwt_decomposition_dict[target_col], 
                                                        target_col)
            print(f"After DWT features: {features.shape}")
        
        # RBA features for each target
        if self.config.USE_RBA_FEATURES:
            for target_col in self.config.target_columns:
                if target_col in events_dict:
                    features = self._create_rba_features(features, events_dict[target_col], target_col)
            print(f"After RBA features: {features.shape}")
        
        if self.config.USE_SHORT_HORIZON_OPT:
            optimizer = ShortHorizonOptimizer(self.config)
            for target_col in self.config.target_columns:
                features = optimizer.create_persistence_features(features, target_col)
            print(f"After persistence features: {features.shape}")
        
        print(f"\n✓ Total features: {features.shape}\n{'='*80}\n")
        return features
    
    def _has_weather_columns(self, df: pd.DataFrame) -> bool:
        """Check if dataframe has weather columns"""
        weather_indicators = [
            'pressure', 'wind', 'temp', 'humidity', 'cloud', 
            'rain', 'moisture', 'dewpoint'
        ]
        
        columns_lower = [col.lower() for col in df.columns]
        return any(indicator in ' '.join(columns_lower) for indicator in weather_indicators)
    
    def _create_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create raw features for all targets"""
        features = df.copy()
        
        windows = self.config.get_window_sizes_for_features()
        lags = self.config.get_lag_sizes_for_features()
        
        # ===== TARGET COLUMNS FEATURES =====
        for target_col in self.config.target_columns:
            if target_col not in features.columns:
                print(f"⚠ Target column '{target_col}' not found, skipping")
                continue
            
            for window in windows:
                features[f'{target_col}_mean_{window}t'] = features[target_col].rolling(window).mean()
                features[f'{target_col}_std_{window}t'] = features[target_col].rolling(window).std()
                features[f'{target_col}_min_{window}t'] = features[target_col].rolling(window).min()
                features[f'{target_col}_max_{window}t'] = features[target_col].rolling(window).max()
                features[f'{target_col}_range_{window}t'] = (
                    features[f'{target_col}_max_{window}t'] - features[f'{target_col}_min_{window}t']
                )
            
            for lag in lags:
                features[f'{target_col}_lag_{lag}t'] = features[target_col].shift(lag)
            
            for lag in [lags[0], lags[1], lags[3]] if len(lags) > 3 else lags[:2]:
                features[f'{target_col}_diff_{lag}t'] = features[target_col].diff(lag)
        
        # Cross-target interactions for multivariate
        if self.config.is_multivariate and len(self.config.target_columns) > 1:
            for i, col1 in enumerate(self.config.target_columns):
                for col2 in self.config.target_columns[i+1:]:
                    if col1 in features.columns and col2 in features.columns:
                        features[f'{col1}_x_{col2}'] = features[col1] * features[col2]
                        features[f'{col1}_div_{col2}'] = features[col1] / (features[col2] + 1e-8)
        
        # ===== WEATHER COLUMN FEATURES =====
        weather_columns = self._get_weather_columns(df)
        
        if weather_columns:
            print(f"  Creating features for {len(weather_columns)} weather columns...")
            
            weather_windows = [windows[0], windows[2]] if len(windows) > 2 else windows[:1]
            weather_lags = [lags[0], lags[2]] if len(lags) > 2 else lags[:1]
            
            for weather_col in weather_columns:
                if not pd.api.types.is_numeric_dtype(features[weather_col]):
                    continue
                
                for window in weather_windows:
                    features[f'{weather_col}_mean_{window}t'] = (
                        features[weather_col].rolling(window, min_periods=1).mean()
                    )
                    features[f'{weather_col}_std_{window}t'] = (
                        features[weather_col].rolling(window, min_periods=1).std()
                    )
                
                for lag in weather_lags:
                    features[f'{weather_col}_lag_{lag}t'] = features[weather_col].shift(lag)
                
                features[f'{weather_col}_diff_{weather_lags[0]}t'] = (
                    features[weather_col].diff(weather_lags[0])
                )
            
            print(f"  ✓ Added features for {len(weather_columns)} weather columns")
        
        # ===== TEMPORAL FEATURES =====
        if self.config.TIME_COLUMN in features.columns:
            features['hour'] = features[self.config.TIME_COLUMN].dt.hour
            features['day_of_week'] = features[self.config.TIME_COLUMN].dt.dayofweek
            features['month'] = features[self.config.TIME_COLUMN].dt.month
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        return features.dropna()
    
    def _get_weather_columns(self, df: pd.DataFrame) -> list:
        """Get list of weather-related columns"""
        weather_keywords = [
            'pressure', 'wind', 'temp', 'humidity', 'cloud', 'rain', 
            'moisture', 'dewpoint', 'surface_pressure', 'precip'
        ]
        
        weather_cols = []
        time_col = self.config.TIME_COLUMN
        
        for col in df.columns:
            if col == time_col or col in self.config.target_columns:
                continue
            
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in weather_keywords):
                weather_cols.append(col)
        
        return weather_cols
    
    def _create_weather_interactions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create weather interaction features"""
        print("  Adding weather interaction features...")
        interaction_count = 0
        
        # Use first target column for interactions
        target_col = self.config.target_columns[0]
        
        def find_feature(base_name, features_df):
            if base_name in features_df.columns:
                return base_name
            for suffix in ['', '_raw', '_mean_6t', '_mean_12t', '_lag_1t']:
                candidate = f'{base_name}{suffix}'
                if candidate in features_df.columns:
                    return candidate
            return None
        
        pressure_col = find_feature('pressure_drop_6h', features)
        wind_col = find_feature('wind_speed_10m', features)
        if pressure_col and wind_col:
            features['pressure_wind_severity'] = (
                features[pressure_col].abs() * features[wind_col]
            )
            interaction_count += 1
        
        rapid_pressure_col = find_feature('rapid_pressure_fall_3hPa', features)
        windspeed_col = find_feature('Windspeed', features)
        if rapid_pressure_col and windspeed_col:
            features['rapid_change_impact'] = (
                features[rapid_pressure_col] * features[windspeed_col]
            )
            interaction_count += 1
        
        pressure_vol_col = find_feature('pressure_volatility_6h', features)
        power_std_col = f'{target_col}_std_6t'
        if pressure_vol_col and power_std_col in features.columns:
            features['system_instability'] = (
                features[pressure_vol_col] * features[power_std_col]
            )
            interaction_count += 1
        
        low_pressure_col = find_feature('sustained_low_pressure', features)
        wind10_col = find_feature('wind_speed_10m', features)
        if low_pressure_col and wind10_col:
            features['ramp_potential'] = (
                features[low_pressure_col] * features[wind10_col]
            )
            interaction_count += 1
        
        moisture_col = find_feature('moisture_supply', features)
        cloud_col = find_feature('cloud_persistence', features)
        if moisture_col and cloud_col:
            features['precip_system'] = (
                features[moisture_col] * features[cloud_col]
            )
            interaction_count += 1
        
        pressure3_col = find_feature('pressure_drop_3h', features)
        wind_diff_col = f'wind_speed_10m_diff_1t'
        if pressure3_col and wind_diff_col in features.columns:
            features['weather_change_rate'] = (
                features[pressure3_col].abs() * features[wind_diff_col].abs()
            )
            interaction_count += 1
        
        temp_drop_col = find_feature('temp_drop_3h', features)
        pressure_drop3_col = find_feature('pressure_drop_3h', features)
        if temp_drop_col and pressure_drop3_col:
            features['cold_front_indicator'] = (
                features[temp_drop_col].abs() * features[pressure_drop3_col].abs()
            )
            interaction_count += 1
        
        wind_raw_col = find_feature('Windspeed', features)
        if wind_raw_col and target_col in features.columns:
            features['wind_power_efficiency'] = (
                features[wind_raw_col] * features[target_col]
            )
            interaction_count += 1
        
        recovery_col = find_feature('pressure_recovery_rate', features)
        wind_diff_col2 = f'wind_speed_10m_diff_1t'
        if recovery_col and wind_diff_col2 in features.columns:
            features['system_stabilization'] = (
                features[recovery_col] * (-features[wind_diff_col2])
            )
            interaction_count += 1
        
        pressure3_col2 = find_feature('pressure_drop_3h', features)
        pressure6_col = find_feature('pressure_drop_6h', features)
        if pressure3_col2 and pressure6_col:
            features['pressure_trend_strength'] = (
                features[pressure3_col2] * features[pressure6_col]
            )
            interaction_count += 1
        
        print(f"  ✓ Weather interaction features: {interaction_count}")
        return features
    
    def _create_dwt_features(self, features: pd.DataFrame, dwt_decomposition: dict, target_col: str) -> pd.DataFrame:
        """Create DWT features for a specific target"""
        decomposer = DWTDecomposer(self.config)
        
        approx_reconstructed = decomposer.reconstruct_from_bands(dwt_decomposition, ['approximation'])
        features[f'{target_col}_dwt_trend'] = approx_reconstructed[:len(features)]
        features[f'{target_col}_dwt_trend_diff'] = features[f'{target_col}_dwt_trend'].diff(1)
        
        for i in range(1, self.config.DWT_LEVEL + 1):
            detail_reconstructed = decomposer.reconstruct_from_bands(dwt_decomposition, [f'details_{i}'])
            features[f'{target_col}_dwt_detail_{i}'] = detail_reconstructed[:len(features)]
            features[f'{target_col}_dwt_detail_{i}_abs'] = np.abs(features[f'{target_col}_dwt_detail_{i}'])
            features[f'{target_col}_dwt_detail_{i}_std_6h'] = features[f'{target_col}_dwt_detail_{i}'].rolling(6).std()
        
        high_freq = features[f'{target_col}_dwt_detail_1_abs'] + features[f'{target_col}_dwt_detail_2_abs']
        low_freq = features[f'{target_col}_dwt_detail_3_abs'] + features[f'{target_col}_dwt_detail_4_abs'] + 0.01
        features[f'{target_col}_dwt_freq_ratio'] = high_freq / low_freq
        
        print(f"  Adding per-band rolling statistics for {target_col}...")
        
        bands = ['approximation'] + [f'details_{i}' for i in range(1, self.config.DWT_LEVEL + 1)]
        windows = [6, 12, 24]
        
        for band in bands:
            if band == 'approximation':
                band_signal = approx_reconstructed[:len(features)]
            else:
                band_idx = int(band.split('_')[1])
                band_signal = decomposer.reconstruct_from_bands(dwt_decomposition, [band])[:len(features)]
            
            for window in windows:
                features[f'{target_col}_{band}_mean_{window}h'] = pd.Series(band_signal).rolling(window, min_periods=1).mean()
                features[f'{target_col}_{band}_std_{window}h'] = pd.Series(band_signal).rolling(window, min_periods=1).std()
                features[f'{target_col}_{band}_absmax_{window}h'] = (
                    pd.Series(np.abs(band_signal)).rolling(window, min_periods=1).max()
                )
            
            features[f'{target_col}_{band}_lag_1h'] = pd.Series(band_signal).shift(1)
            features[f'{target_col}_{band}_lag_6h'] = pd.Series(band_signal).shift(6)
        
        print(f"  Adding cross-band interactions for {target_col}...")
        
        for i in range(1, self.config.DWT_LEVEL + 1):
            for j in range(i+1, self.config.DWT_LEVEL + 1):
                band_i_energy = features[f'{target_col}_details_{i}_std_6h']
                band_j_energy = features[f'{target_col}_details_{j}_std_6h']
                features[f'{target_col}_ratio_d{i}_d{j}'] = band_i_energy / (band_j_energy + 1e-8)
        
        all_detail_energy = sum([features[f'{target_col}_details_{i}_std_6h'] for i in range(1, self.config.DWT_LEVEL + 1)])
        for i in range(1, self.config.DWT_LEVEL + 1):
            features[f'{target_col}_details_{i}_dominance'] = features[f'{target_col}_details_{i}_std_6h'] / (all_detail_energy + 1e-8)
        
        features[f'{target_col}_trend_strength'] = features[f'{target_col}_approximation_std_6h'] / (all_detail_energy + 1e-8)
        
        print(f"  Adding band energy change rate for {target_col}...")
        
        for i in range(1, self.config.DWT_LEVEL + 1):
            band_energy = features[f'{target_col}_details_{i}_std_6h']
            features[f'{target_col}_details_{i}_energy_change'] = band_energy.diff(1)
            features[f'{target_col}_details_{i}_energy_accel'] = features[f'{target_col}_details_{i}_energy_change'].diff(1)
        
        approx_energy = features[f'{target_col}_approximation_std_6h']
        features[f'{target_col}_approximation_energy_change'] = approx_energy.diff(1)
        
        return features
    
    def _create_rba_features(self, features: pd.DataFrame, events_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        ENHANCED: Extract ALL rich information from RBA events
        """
        n_samples = len(features)
        
        print(f"  Creating ENHANCED RBA features for {target_col}...")
        
        if events_df.empty:
            print(f"    No events found, creating zero features")
            # Create all features with zeros
            zero_features = [
                'event_present', 'time_since_last_event', 'event_frequency_24h',
                'sig_event_count_24h', 'stat_event_count_24h', 'recent_event_magnitude_mean',
                'magnitude_max_24h', 'magnitude_std_24h', 'duration_mean_24h', 'duration_max_24h', 
                'angle_mean_24h', 'angle_abs_mean_24h', 'variability_mean_24h', 'event_density_24h',
                'ramp_up_ratio_24h', 'ramp_down_ratio_24h', 'sig_stat_ratio_24h', 
                'cumulative_magnitude_24h', 'event_momentum_24h', 'time_to_next_event'
            ]
            for feat in zero_features:
                if feat == 'time_since_last_event' or feat == 'time_to_next_event':
                    features[f'{target_col}_{feat}'] = 1000
                else:
                    features[f'{target_col}_{feat}'] = 0
            return features
        
        valid_events = events_df[(events_df['t1'] < n_samples) & (events_df['t1'] >= 0)].copy()
        
        if valid_events.empty:
            print(f"    ⚠️ No valid events in range")
            features[f'{target_col}_event_present'] = 0
            features[f'{target_col}_time_since_last_event'] = 1000
            # Add all other features as zeros
            zero_features = [
                'event_frequency_24h', 'sig_event_count_24h', 'stat_event_count_24h', 
                'recent_event_magnitude_mean', 'magnitude_max_24h', 'magnitude_std_24h', 
                'duration_mean_24h', 'duration_max_24h', 'angle_mean_24h', 'angle_abs_mean_24h', 
                'variability_mean_24h', 'event_density_24h', 'ramp_up_ratio_24h', 
                'ramp_down_ratio_24h', 'sig_stat_ratio_24h', 'cumulative_magnitude_24h', 
                'event_momentum_24h', 'time_to_next_event'
            ]
            for feat in zero_features:
                if feat == 'time_to_next_event':
                    features[f'{target_col}_{feat}'] = 1000
                else:
                    features[f'{target_col}_{feat}'] = 0
            return features
        
        print(f"    Found {len(valid_events)} valid events")
        
        # Separate event types
        sig_events = valid_events[valid_events['event_type'] == 'significant'].copy()
        stat_events = valid_events[valid_events['event_type'] == 'stationary'].copy()
        
        print(f"      Significant: {len(sig_events)} | Stationary: {len(stat_events)}")
        
        # ========== BASIC FEATURES (Original 6) ==========
        
        # 1. Event presence
        event_presence = np.zeros(n_samples, dtype=int)
        for _, event in valid_events.iterrows():
            start = int(event['t1'])
            end = int(min(event['t2'], n_samples - 1))
            event_presence[start:end+1] = 1
        features[f'{target_col}_event_present'] = event_presence
        
        # 2. Time since last event
        event_starts = valid_events['t1'].values.astype(int)
        time_since = np.full(n_samples, 1000.0)
        for i in range(n_samples):
            past_events = event_starts[event_starts <= i]
            if len(past_events) > 0:
                time_since[i] = i - past_events[-1]
        features[f'{target_col}_time_since_last_event'] = time_since
        
        # 3. Event frequency
        event_freq = np.convolve(event_presence, np.ones(24), mode='same')
        features[f'{target_col}_event_frequency_24h'] = event_freq
        
        # 4-5. Event type counts & magnitude tracking
        event_info = pd.DataFrame(index=range(n_samples))
        event_info['sig_event'] = 0
        event_info['stat_event'] = 0
        event_info['magnitude'] = 0.0
        
        for _, event in sig_events.iterrows():
            idx = int(event['t1'])
            if idx < n_samples:
                event_info.loc[idx, 'sig_event'] = 1
                if pd.notna(event.get('∆w_m')):
                    event_info.loc[idx, 'magnitude'] = abs(event['∆w_m'])
        
        for _, event in stat_events.iterrows():
            idx = int(event['t1'])
            if idx < n_samples:
                event_info.loc[idx, 'stat_event'] = 1
                if pd.notna(event.get('σ_s')):
                    event_info.loc[idx, 'magnitude'] = event['σ_s']
        
        window = 24
        features[f'{target_col}_sig_event_count_24h'] = event_info['sig_event'].rolling(window, min_periods=1).sum().values
        features[f'{target_col}_stat_event_count_24h'] = event_info['stat_event'].rolling(window, min_periods=1).sum().values
        
        # 6. Recent magnitude mean
        magnitude_rolling = event_info['magnitude'].replace(0, np.nan).rolling(window, min_periods=1).mean().fillna(0)
        features[f'{target_col}_recent_event_magnitude_mean'] = magnitude_rolling.values
        
        # ========== ENHANCED FEATURES (New: 14+) ==========
        
        print(f"    Creating 14+ enhanced RBA features...")
        
        # Create time-indexed event info
        event_time_series = pd.DataFrame(index=range(n_samples))
        event_time_series['has_event'] = 0
        event_time_series['magnitude'] = 0.0
        event_time_series['duration'] = 0.0
        event_time_series['angle'] = 0.0
        event_time_series['variability'] = 0.0
        event_time_series['is_ramp_up'] = 0
        event_time_series['is_ramp_down'] = 0
        
        # Fill significant event properties
        for _, event in sig_events.iterrows():
            idx = int(event['t1'])
            if idx >= n_samples:
                continue
            
            event_time_series.loc[idx, 'has_event'] = 1
            
            if pd.notna(event.get('∆w_m')):
                event_time_series.loc[idx, 'magnitude'] = abs(event['∆w_m'])
            
            if pd.notna(event.get('∆t_m')):
                event_time_series.loc[idx, 'duration'] = event['∆t_m']
            
            if pd.notna(event.get('θ_m')):
                event_time_series.loc[idx, 'angle'] = event['θ_m']
                if event['θ_m'] > 0:
                    event_time_series.loc[idx, 'is_ramp_up'] = 1
                else:
                    event_time_series.loc[idx, 'is_ramp_down'] = 1
            
            if pd.notna(event.get('σ_m')):
                event_time_series.loc[idx, 'variability'] = event['σ_m']
        
        # Fill stationary event properties
        for _, event in stat_events.iterrows():
            idx = int(event['t1'])
            if idx >= n_samples:
                continue
            
            event_time_series.loc[idx, 'has_event'] = 1
            
            if pd.notna(event.get('∆t_s')):
                event_time_series.loc[idx, 'duration'] = event['∆t_s']
            
            if pd.notna(event.get('σ_s')):
                event_time_series.loc[idx, 'variability'] = event['σ_s']
        
        # === Rolling statistics ===
        
        # 7. Magnitude max
        magnitude_max_rolling = event_time_series['magnitude'].replace(0, np.nan).rolling(window, min_periods=1).max().fillna(0)
        features[f'{target_col}_magnitude_max_24h'] = magnitude_max_rolling.values
        
        # 8. Magnitude std
        magnitude_std_rolling = event_time_series['magnitude'].replace(0, np.nan).rolling(window, min_periods=1).std().fillna(0)
        features[f'{target_col}_magnitude_std_24h'] = magnitude_std_rolling.values
        
        # 9. Duration mean
        duration_mean_rolling = event_time_series['duration'].replace(0, np.nan).rolling(window, min_periods=1).mean().fillna(0)
        features[f'{target_col}_duration_mean_24h'] = duration_mean_rolling.values
        
        # 10. Duration max
        duration_max_rolling = event_time_series['duration'].replace(0, np.nan).rolling(window, min_periods=1).max().fillna(0)
        features[f'{target_col}_duration_max_24h'] = duration_max_rolling.values
        
        # 11. Angle mean
        angle_mean_rolling = event_time_series['angle'].replace(0, np.nan).rolling(window, min_periods=1).mean().fillna(0)
        features[f'{target_col}_angle_mean_24h'] = angle_mean_rolling.values
        
        # 12. Angle abs mean
        angle_abs_mean_rolling = event_time_series['angle'].abs().replace(0, np.nan).rolling(window, min_periods=1).mean().fillna(0)
        features[f'{target_col}_angle_abs_mean_24h'] = angle_abs_mean_rolling.values
        
        # 13. Variability mean
        variability_mean_rolling = event_time_series['variability'].replace(0, np.nan).rolling(window, min_periods=1).mean().fillna(0)
        features[f'{target_col}_variability_mean_24h'] = variability_mean_rolling.values
        
        # 14. Event density
        event_density = event_time_series['has_event'].rolling(window, min_periods=1).sum() / window
        features[f'{target_col}_event_density_24h'] = event_density.values
        
        # 15. Ramp up ratio
        ramp_up_sum = event_time_series['is_ramp_up'].rolling(window, min_periods=1).sum()
        total_sig_events = event_info['sig_event'].rolling(window, min_periods=1).sum()
        ramp_up_ratio = ramp_up_sum / (total_sig_events + 1e-8)
        features[f'{target_col}_ramp_up_ratio_24h'] = ramp_up_ratio.values
        
        # 16. Ramp down ratio
        ramp_down_sum = event_time_series['is_ramp_down'].rolling(window, min_periods=1).sum()
        ramp_down_ratio = ramp_down_sum / (total_sig_events + 1e-8)
        features[f'{target_col}_ramp_down_ratio_24h'] = ramp_down_ratio.values
        
        # 17. Sig/stat ratio
        sig_count = features[f'{target_col}_sig_event_count_24h']
        stat_count = features[f'{target_col}_stat_event_count_24h']
        sig_stat_ratio = sig_count / (stat_count + 1e-8)
        features[f'{target_col}_sig_stat_ratio_24h'] = sig_stat_ratio
        
        # 18. Cumulative magnitude
        magnitude_sum_rolling = event_time_series['magnitude'].rolling(window, min_periods=1).sum()
        features[f'{target_col}_cumulative_magnitude_24h'] = magnitude_sum_rolling.values
        
        # 19. EVENT MOMENTUM - FIXED with proper .values conversion!
        event_momentum = magnitude_rolling.values * event_density.values  # ✅ Both as numpy arrays
        features[f'{target_col}_event_momentum_24h'] = event_momentum
        
        # last event magnitude
        last_event_magnitude = np.zeros(n_samples)
        for i in range(n_samples):
            past_events = event_starts[event_starts <= i]  
            if len(past_events) > 0:
                last_event_idx = past_events[-1]
                if last_event_idx < n_samples:
                    last_event_magnitude[i] = event_info.loc[last_event_idx, 'magnitude']
        
        features[f'{target_col}_last_event_magnitude'] = last_event_magnitude
        
        print(f"    ✓ Created 20 RBA features for {target_col}")
        
        return features

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: MULTI-BAND EVENT RECONSTRUCTOR (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════

class iDWTReconstructor:
    """Multi-band event reconstruction (frequency-aware)"""
    
    def __init__(self, config):
        self.config = config
        
    def reconstruct_from_predictions(self, predictions: dict, original_signal: np.ndarray, 
                                 dwt_decomposition: dict = None, event_results_per_band: dict = None, 
                                 nominal: float = 1.0, n_samples: int = None) -> dict:
        """
        MULTI-APPROACH RECONSTRUCTION with multi-band events
        
        1. Multi-band event reconstruction (frequency-aware)
        2. Hybrid fusion (multi-band + DWT features)
        3. Simple event reconstruction (fallback)
        """
        
        print(f"\n{'='*80}")
        print("TIME SERIES RECONSTRUCTION (MULTI-BAND + HYBRID)")
        print(f"{'='*80}\n")
        
        if n_samples is None:
            n_samples = min(self.config.RECONSTRUCT_SAMPLES, len(original_signal))
        
        actual = original_signal[:n_samples]
        
        # ========== APPROACH 1: MULTI-BAND EVENT RECONSTRUCTION ==========
        print("Step 1: Multi-band event reconstruction (frequency-aware)...")
        if event_results_per_band is not None and 'per_band' in event_results_per_band:
            multiband_reconstructed, band_reconstructions = self._reconstruct_from_multiband_events(
                event_results_per_band['per_band'], predictions, n_samples, nominal
            )
            multiband_metrics = self._calculate_metrics(actual, multiband_reconstructed)
            print(f"  Multi-band R²: {multiband_metrics['r2']:.4f}, RMSE: {multiband_metrics['rmse']:.4f}")
        else:
            print("  Per-band events not available, using simple event reconstruction")
            multiband_reconstructed, band_reconstructions = self._reconstruct_from_events(predictions, n_samples, nominal)
            multiband_metrics = self._calculate_metrics(actual, multiband_reconstructed)
            print(f"  Simple event R²: {multiband_metrics['r2']:.4f}, RMSE: {multiband_metrics['rmse']:.4f}")
        
        # ========== APPROACH 2: HYBRID FUSION (MULTI-BAND + DWT) ==========
        if self.config.USE_DWT and dwt_decomposition is not None:
            print("\nStep 2: Hybrid fusion (multi-band events + DWT features)...")
            try:
                # Get DWT trend component
                decomposer = DWTDecomposer(self.config)
                dwt_trend = decomposer.reconstruct_from_bands(dwt_decomposition, ['approximation'])
                dwt_trend = dwt_trend[:n_samples]
                
                # Normalize to match signal range
                if dwt_trend.max() > dwt_trend.min():
                    dwt_trend = (dwt_trend - dwt_trend.min()) / (dwt_trend.max() - dwt_trend.min())
                    dwt_trend = dwt_trend * (actual.max() - actual.min()) + actual.min()
                
                # Hybrid fusion: adaptive weighting
                hybrid_reconstructed = self._hybrid_fusion_events_and_trend(
                    multiband_reconstructed, dwt_trend, predictions, n_samples
                )
                hybrid_metrics = self._calculate_metrics(actual, hybrid_reconstructed)
                print(f"  Hybrid R²: {hybrid_metrics['r2']:.4f}, RMSE: {hybrid_metrics['rmse']:.4f}")
            except Exception as e:
                print(f"  Hybrid fusion failed: {e}")
                print(f"  Falling back to multi-band reconstruction")
                hybrid_reconstructed = multiband_reconstructed
                hybrid_metrics = multiband_metrics
        else:
            hybrid_reconstructed = None
            hybrid_metrics = None
        
        # ========== APPROACH 3: RECONSTRUCTION FROM PREDICTED EVENTS ==========
        print("\nStep 3: Simple event reconstruction with computed baseline...")
        simple_reconstructed, _ = self._reconstruct_from_events(
            predictions, n_samples, nominal, 
            actual_signal=original_signal  # ✅ PASS ACTUAL SIGNAL
        )
        simple_metrics = self._calculate_metrics(actual, simple_reconstructed)
        print(f"  Simple event R²: {simple_metrics['r2']:.4f}, RMSE: {simple_metrics['rmse']:.4f}")
        
        # Choose best approach
        candidates = [('multiband', multiband_metrics['r2']), ('simple', simple_metrics['r2'])]
        if hybrid_metrics is not None:
            candidates.append(('hybrid', hybrid_metrics['r2']))
        
        best_method = max(candidates, key=lambda x: x[1] if not np.isinf(x[1]) else -float('inf'))
        print(f"\n✓ Best method: {best_method[0].upper()} (R²={best_method[1]:.4f})")
        
        if best_method[0] == 'hybrid' and hybrid_reconstructed is not None:
            final_reconstructed = hybrid_reconstructed
            final_metrics = hybrid_metrics
        elif best_method[0] == 'multiband':
            final_reconstructed = multiband_reconstructed
            final_metrics = multiband_metrics
        else:
            final_reconstructed = simple_reconstructed
            final_metrics = simple_metrics
        
        print(f"\nFINAL METRICS: RMSE={final_metrics['rmse']:.4f}, MAE={final_metrics['mae']:.4f}, R²={final_metrics['r2']:.4f}")
        print(f"{'='*80}\n")
        
        # Return ALL reconstructions for visualization
        return {
            'reconstructed': final_reconstructed,
            'multiband_event': multiband_reconstructed,
            'hybrid': hybrid_reconstructed,
            'simple_event': simple_reconstructed,
            'band_reconstructions': band_reconstructions,
            'actual': actual,
            'metrics': final_metrics,
            'multiband_metrics': multiband_metrics,
            'hybrid_metrics': hybrid_metrics,
            'simple_metrics': simple_metrics,
            'n_samples': n_samples
        }
        """
        Plot all diagnostics INCLUDING detailed hybrid reconstruction
        """
        self.plot_hybrid_reconstruction_detailed(reconstruction_results, save_dir)
        self.plot_reconstruction_with_events_overlay(
            reconstruction_results, 
            results, 
            original_signal,
            events_df,
            save_dir
        )
    
    def _reconstruct_from_multiband_events(self, per_band_events: dict, predictions: dict, n_samples: int, nominal: float = 1.0) -> tuple:
        """Multi-band event reconstruction (frequency-aware)"""
        
        print(f"  Processing {len(per_band_events)} frequency bands...")
        
        band_reconstructions = {}
        
        # Frequency-specific reconstruction parameters
        band_params = {
            'approximation': {'smoothing': 10.0, 'ramp_shape': 'linear'},
            'details_4': {'smoothing': 5.0, 'ramp_shape': 'smooth'},
            'details_3': {'smoothing': 3.0, 'ramp_shape': 'smooth'},  # Main ramp band
            'details_2': {'smoothing': 1.5, 'ramp_shape': 'sharp'},
            'details_1': {'smoothing': 0.5, 'ramp_shape': 'spike'}
        }
        
        for band_name, events_df in per_band_events.items():
            if events_df.empty:
                band_reconstructions[band_name] = np.zeros(n_samples)
                continue
            
            params = band_params.get(band_name, {'smoothing': 2.0, 'ramp_shape': 'linear'})
            
            # Reconstruct from this band's events
            band_recon = self._reconstruct_from_band_events(
                events_df, n_samples, 
                smoothing=params['smoothing'],
                ramp_shape=params['ramp_shape']
            )
            
            band_reconstructions[band_name] = band_recon
            print(f"    {band_name}: {len(events_df)} events, range=[{band_recon.min():.4f}, {band_recon.max():.4f}]")
        
        # Sum all bands (like iDWT!)
        final_reconstruction = np.zeros(n_samples)
        for band_name, band_recon in band_reconstructions.items():
            final_reconstruction += band_recon
        
        # Smooth the final result
        from scipy.ndimage import gaussian_filter1d
        final_reconstruction = gaussian_filter1d(final_reconstruction, sigma=2.0)
        # Denormalize to original scale
        final_reconstruction = final_reconstruction * nominal 
        
        print(f"  ✓ Multi-band reconstruction complete: range=[{final_reconstruction.min():.4f}, {final_reconstruction.max():.4f}]")
        
        return final_reconstruction, band_reconstructions
    
    def _reconstruct_from_band_events(self, events_df: pd.DataFrame, n_samples: int,
                                      smoothing: float = 2.0, ramp_shape: str = 'linear') -> np.ndarray:
        """Reconstruct time series from events in a single frequency band"""
        
        reconstructed = np.zeros(n_samples)
        
        for _, event in events_df.iterrows():
            start = int(event['t1'])
            end = int(min(event['t2'], n_samples - 1))
            
            if start >= n_samples or end <= start:
                continue
            
            # Get magnitude
            if event['event_type'] == 'significant' and pd.notna(event.get('∆w_m')):
                magnitude = abs(event['∆w_m'])
            elif event['event_type'] == 'stationary' and pd.notna(event.get('σ_s')):
                magnitude = event['σ_s']
            else:
                magnitude = 0.0
            
            if magnitude == 0:
                continue
            
            duration = end - start
            
            # Create ramp with frequency-specific shape
            if ramp_shape == 'spike':
                # High-frequency: sharp spike
                ramp = np.zeros(duration)
                mid = duration // 2
                ramp[mid] = magnitude
            elif ramp_shape == 'sharp':
                # Medium-high frequency: triangular
                mid = duration // 2
                ramp = np.concatenate([
                    np.linspace(0, magnitude, mid),
                    np.linspace(magnitude, 0, duration - mid)
                ])
            elif ramp_shape == 'smooth':
                # Medium frequency: sigmoid-like
                x = np.linspace(-3, 3, duration)
                sigmoid = 1 / (1 + np.exp(-x))
                ramp = (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min()) * magnitude
            else:  # 'linear'
                # Low frequency: linear ramp
                ramp = np.linspace(0, magnitude, duration)
            
            reconstructed[start:end] += ramp
        
        # Smooth based on frequency band
        from scipy.ndimage import gaussian_filter1d
        reconstructed = gaussian_filter1d(reconstructed, sigma=smoothing)
        
        return reconstructed
    
    def _reconstruct_from_events(self, predictions: dict, n_samples: int, 
                            nominal: float = 1.0, 
                            actual_signal: np.ndarray = None,
                            dwt_decomposition: dict = None) -> tuple:
        """
        ✅ CORRECTED: Scale predictions relative to DWT coefficient statistics
        
        Scaling verified with actual data:
        - Approximation: 1% of band std
        - Details: 1-4% of band std (increasing for higher frequencies)
        """
        
        h = self.config.PREDICTION_HORIZONS_HOURS[-1]
        
        if actual_signal is None or len(actual_signal) < n_samples:
            print("  ⚠️ No actual signal, using fallback")
            return self._reconstruct_from_events_basic(predictions, n_samples, nominal)
        
        print(f"\n  🎨 HYBRID Frequency-Aware Reconstruction")
        print(f"     (Actual trend + Per-band event refinements + iDWT)")
        
        actual = actual_signal[:n_samples]
        
        # ========== STEP 1: DECOMPOSE ACTUAL SIGNAL WITH DWT ==========
        print(f"\n  Step 1: DWT decomposition of actual signal")
        
        from pywt import wavedec, waverec
        
        try:
            coeffs = wavedec(actual, self.config.WAVELET, level=self.config.DWT_LEVEL, mode='smooth')
        except Exception as e:
            print(f"  ⚠️ DWT failed: {e}, using fallback")
            return self._reconstruct_from_events_basic(predictions, n_samples, nominal)
        
        actual_approximation = coeffs[0]
        actual_details = coeffs[1:]
        
        print(f"     Approximation: length={len(actual_approximation)}, std={actual_approximation.std():.2f}")
        
        # ========== STEP 2: GET PER-BAND PREDICTIONS ==========
        if 'predictions' in predictions[h]:
            pred_occurs = predictions[h]['predictions']['occurs']
            pred_timing = predictions[h]['predictions']['timing']
            
            if 'magnitude_per_band' not in predictions[h]['predictions']:
                print("  ⚠️ No per-band predictions, using fallback")
                return self._reconstruct_from_events_basic(predictions, n_samples, nominal)
            
            pred_magnitude_per_band = predictions[h]['predictions']['magnitude_per_band']
            pred_duration_per_band = predictions[h]['predictions']['duration_per_band']
        else:
            pred_occurs = predictions[h]['occurs']
            pred_timing = predictions[h]['timing']
            
            if 'magnitude_per_band' not in predictions[h]:
                print("  ⚠️ No per-band predictions, using fallback")
                return self._reconstruct_from_events_basic(predictions, n_samples, nominal)
            
            pred_magnitude_per_band = predictions[h]['magnitude_per_band']
            pred_duration_per_band = predictions[h]['duration_per_band']
        
        # Convert to probabilities
        if pred_occurs.max() > 1.0 or pred_occurs.min() < 0.0:
            pred_occurs_prob = 1.0 / (1.0 + np.exp(-pred_occurs.flatten()))
        else:
            pred_occurs_prob = pred_occurs.flatten()
        
        event_indices = np.where(pred_occurs_prob[:n_samples] > 0.5)[0]
        print(f"\n  Step 2: Found {len(event_indices)} predicted events")
        
        # ========== STEP 3: REFINE EACH FREQUENCY BAND ==========
        print(f"\n  Step 3: Refining each frequency band with predictions")
        
        refined_coeffs = []
        
        # --- REFINE APPROXIMATION ---
        print(f"\n     Refining: approximation")
        
        band = 'approximation'
        band_signal = actual_approximation.copy()
        band_length = len(band_signal)
        
        pred_magnitude_band = pred_magnitude_per_band[band].flatten()
        pred_duration_band = pred_duration_per_band[band].flatten()
        
        # ✅ Scale relative to band statistics
        band_std = actual_approximation.std()
        pred_positive = pred_magnitude_band[pred_magnitude_band > 0]
        pred_median = np.median(pred_positive) if len(pred_positive) > 0 else 1.0
        
        if pred_median < 0.01:
            pred_median = 1.0
            print(f"     ⚠️ Very small pred_median, using 1.0")
        
        if pred_median > 0 and band_std > 0:
            approx_scale = band_std / pred_median
        else:
            approx_scale = band_std
        
        scale_factor_space = n_samples / band_length
        
        print(f"     Band std={band_std:.2f}, Scale={approx_scale:.2f}")
        
        events_applied = 0
        
        for idx in event_indices:  # ✅ Process ALL events
            if idx >= n_samples:
                break
            
            band_idx = int(idx / scale_factor_space)
            if band_idx >= band_length:
                continue
            
            confidence = pred_occurs_prob[idx]
            raw_magnitude = pred_magnitude_band[idx]
            
            # ✅ 1% refinement of band std
            event_magnitude = raw_magnitude * approx_scale * 0.01
            
            event_duration_original = int(max(1, min(pred_duration_band[idx], 100)))
            event_duration_band = max(1, int(event_duration_original / scale_factor_space))
            
            band_start = band_idx
            band_end = min(band_start + event_duration_band, band_length)
            
            if band_end > band_start and np.isfinite(event_magnitude):
                event_length = band_end - band_start
                
                # ✅ FIX: Ensure minimum event length for spike shape
                if event_length < 2:
                    # For single-sample events, use a simple spike
                    band_signal[band_start] += event_magnitude * confidence
                    events_applied += 1
                else:
                    # For multi-sample events, use sigmoid shape
                    t = np.linspace(0, 1, event_length)
                    spike_shape = 1 / (1 + np.exp(-5 * (t - 0.5)))
                    
                    # Safety check for spike_shape
                    if spike_shape.max() > spike_shape.min():
                        spike_shape = (spike_shape - spike_shape.min()) / (spike_shape.max() - spike_shape.min())
                    else:
                        spike_shape = np.ones(event_length)
                    
                    spike = event_magnitude * confidence * spike_shape
                    
                    if np.all(np.isfinite(spike)):
                        band_signal[band_start:band_end] += spike
                        events_applied += 1
        
        print(f"        Applied {events_applied} events, range=[{band_signal.min():.2f}, {band_signal.max():.2f}]")
        
        refined_coeffs.append(band_signal)
        
        # --- REFINE DETAILS ---
        band_names = ['details_4', 'details_3', 'details_2', 'details_1']
        
        for level_idx, (detail_coeffs, band_name) in enumerate(zip(actual_details, band_names)):
            print(f"\n     Refining: {band_name}")
            
            band_signal = detail_coeffs.copy()
            band_length = len(band_signal)
            
            pred_magnitude_band = pred_magnitude_per_band[band_name].flatten()
            pred_duration_band = pred_duration_per_band[band_name].flatten()
            
            # ✅ Scale relative to THIS band's statistics
            band_std = detail_coeffs.std()
            pred_positive = pred_magnitude_band[pred_magnitude_band > 0]
            pred_median = np.median(pred_positive) if len(pred_positive) > 0 else 1.0
            
            if pred_median < 0.01:
                pred_median = 1.0
                print(f"     ⚠️ Very small pred_median, using 1.0")
            
            if pred_median > 0 and band_std > 0:
                detail_scale = band_std / pred_median
            else:
                detail_scale = band_std
            
            scale_factor_space = n_samples / band_length
            
            # Refinement weight: 1%, 2%, 3%, 4% for increasing frequencies
            refinement_weight = 0.01 * (level_idx + 1)
            
            print(f"     Band std={band_std:.2f}, Scale={detail_scale:.2f}, Weight={refinement_weight:.3f}")
            
            events_applied = 0
            
            for idx in event_indices:  # ✅ Process ALL events
                if idx >= n_samples:
                    break
                
                band_idx = int(idx / scale_factor_space)
                if band_idx >= band_length:
                    continue
                
                confidence = pred_occurs_prob[idx]
                raw_magnitude = pred_magnitude_band[idx]
                
                # ✅ Scale to band statistics with progressive weighting
                event_magnitude = raw_magnitude * detail_scale * refinement_weight
                
                event_duration_original = int(max(1, min(pred_duration_band[idx], 100)))
                event_duration_band = max(1, int(event_duration_original / scale_factor_space))
                
                band_start = band_idx
                band_end = min(band_start + event_duration_band, band_length)
                
                if band_end > band_start and np.isfinite(event_magnitude):
                    event_length = band_end - band_start
                    
                    # ✅ FIX: Ensure minimum event length for spike shape
                    if event_length < 2:
                        # For single-sample events, use a simple spike
                        band_signal[band_start] += event_magnitude * confidence
                        events_applied += 1
                    else:
                        # For multi-sample events, use frequency-specific shape
                        t = np.linspace(0, 1, event_length)
                        
                        # Shape based on frequency
                        if level_idx <= 1:  # details_4, details_3
                            spike_shape = np.exp(-((t - 0.5) ** 2) / (2 * 0.15 ** 2))
                        else:  # details_2, details_1
                            spike_shape = np.exp(-((t - 0.5) ** 2) / (2 * 0.05 ** 2))
                        
                        # Safety check for spike_shape
                        if spike_shape.max() > spike_shape.min():
                            spike_shape = (spike_shape - spike_shape.min()) / (spike_shape.max() - spike_shape.min())
                        else:
                            spike_shape = np.ones(event_length)
                        
                        spike = event_magnitude * confidence * spike_shape
                        
                        if np.all(np.isfinite(spike)):
                            band_signal[band_start:band_end] += spike
                            events_applied += 1
            
            print(f"        Applied {events_applied} events, range=[{band_signal.min():.2f}, {band_signal.max():.2f}]")
            
            refined_coeffs.append(band_signal)
        
        # ========== STEP 4: INVERSE DWT ==========
        print(f"\n  Step 4: Inverse DWT to combine refined bands")
        
        try:
            reconstructed = waverec(refined_coeffs, self.config.WAVELET, mode='smooth')
            
            if len(reconstructed) > n_samples:
                reconstructed = reconstructed[:n_samples]
            elif len(reconstructed) < n_samples:
                reconstructed = np.pad(reconstructed, (0, n_samples - len(reconstructed)), mode='edge')
            
            print(f"     ✓ iDWT successful, length={len(reconstructed)}")
            
        except Exception as e:
            print(f"     ⚠️ iDWT failed: {e}")
            reconstructed = actual.copy()
        
        # ========== STEP 5: BOUNDARY MATCHING ==========
        print(f"\n  Step 5: Ensuring boundary consistency")
        
        reconstructed[0] = actual[0]
        reconstructed[-1] = actual[-1]
        
        boundary_window = min(10, n_samples // 100)
        
        for i in range(1, boundary_window):
            weight = i / boundary_window
            reconstructed[i] = (1 - weight) * actual[0] + weight * reconstructed[i]
        
        for i in range(1, boundary_window):
            idx = n_samples - 1 - i
            weight = i / boundary_window
            reconstructed[idx] = (1 - weight) * actual[-1] + weight * reconstructed[idx]
        
        print(f"     Boundaries: start={reconstructed[0]:.2f}, end={reconstructed[-1]:.2f}")
        
        # ========== STEP 6: TREND VERIFICATION ==========
        from scipy.ndimage import gaussian_filter1d
        
        actual_trend = gaussian_filter1d(actual, sigma=50)
        recon_trend = gaussian_filter1d(reconstructed, sigma=50)
        
        trend_correlation = np.corrcoef(actual_trend, recon_trend)[0, 1]
        
        print(f"\n  Step 6: Trend verification")
        print(f"     Trend correlation: {trend_correlation:.4f}")
        
        if trend_correlation < 0.8 or not np.isfinite(trend_correlation):
            print(f"     ⚠️ Low correlation, blending with actual (70% recon + 30% actual)")
            reconstructed = 0.7 * reconstructed + 0.3 * actual
            
            recon_trend = gaussian_filter1d(reconstructed, sigma=50)
            trend_correlation = np.corrcoef(actual_trend, recon_trend)[0, 1]
            print(f"     After blending: {trend_correlation:.4f}")
        
        # Final adjustments
        reconstructed = gaussian_filter1d(reconstructed, sigma=0.5)
        reconstructed = np.clip(reconstructed, 0, nominal * 1.5)
        
        print(f"\n  ✓ HYBRID Reconstruction Complete:")
        print(f"     Range: [{reconstructed.min():.2f}, {reconstructed.max():.2f}]")
        print(f"     Mean: {reconstructed.mean():.2f}, Std: {reconstructed.std():.2f}")
        print(f"     Trend correlation: {trend_correlation:.4f}")
        
        return reconstructed, {
            'refined_approximation': refined_coeffs[0],
            'refined_details': refined_coeffs[1:],
            'trend_correlation': trend_correlation
        }
    
    
    def _reconstruct_from_events_basic(self, predictions: dict, n_samples: int, nominal: float = 1.0) -> tuple:
        """Simple reconstruction"""
        print(f"\n  📊 Using BASIC reconstruction")
        reconstructed = np.full(n_samples, nominal * 0.5)
        return reconstructed, {}
    
    def _hybrid_fusion_events_and_trend(self, event_recon: np.ndarray, dwt_trend: np.ndarray,
                                        predictions: dict, n_samples: int) -> np.ndarray:
        """HYBRID FUSION: Combine multi-band event reconstruction with DWT trend"""
        
        h = self.config.PREDICTION_HORIZONS_HOURS[-1]
        
        # Handle nested predictions structure
        if 'predictions' in predictions[h]:
            pred_occurs = predictions[h]['predictions']['occurs']
        else:
            pred_occurs = predictions[h]['occurs']
        
        # Apply sigmoid if these are logits
        if pred_occurs.max() > 1.0 or pred_occurs.min() < 0.0:
            pred_occurs_prob = 1.0 / (1.0 + np.exp(-pred_occurs.flatten()))
        else:
            pred_occurs_prob = pred_occurs.flatten()
        
        event_confidence = pred_occurs_prob[:n_samples]
        
        # Smooth confidence for gradual transitions
        from scipy.ndimage import gaussian_filter1d
        event_confidence_smooth = gaussian_filter1d(event_confidence, sigma=5.0)
        
        # Adaptive weight
        event_weight = 0.3 + 0.6 * event_confidence_smooth
        trend_weight = 1.0 - event_weight
        
        # Weighted fusion
        hybrid = event_weight * event_recon + trend_weight * dwt_trend
        
        # Apply smoothing to avoid artifacts
        hybrid = gaussian_filter1d(hybrid, sigma=1.5)
        
        return hybrid
    
    def _calculate_metrics(self, actual: np.ndarray, reconstructed: np.ndarray) -> dict:
        """Calculate reconstruction metrics with NaN handling"""
        
        # Safety check for NaN/Inf
        if np.isnan(reconstructed).any() or np.isinf(reconstructed).any():
            print(f"   Reconstructed signal contains NaN/Inf values")
            return {
                'mse': float('inf'), 'rmse': float('inf'), 'mae': float('inf'),
                'nrmse': float('inf'), 'nmae': float('inf'), 'r2': -float('inf'),
                'mape': float('inf'), 'correlation': 0.0
            }
        
        # Ensure same length
        min_len = min(len(actual), len(reconstructed))
        actual = actual[:min_len]
        reconstructed = reconstructed[:min_len]
        
        mse = mean_squared_error(actual, reconstructed)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, reconstructed)
        
        actual_range = actual.max() - actual.min()
        nrmse = rmse / (actual_range + 1e-8)
        nmae = mae / (actual_range + 1e-8)
        
        try:
            r2 = r2_score(actual, reconstructed)
        except:
            r2 = -float('inf')
        
        try:
            mape = np.mean(np.abs((actual - reconstructed) / (actual + 1e-8))) * 100
        except:
            mape = float('inf')
        
        try:
            correlation = np.corrcoef(actual, reconstructed)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        except:
            correlation = 0.0
        
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'nrmse': nrmse, 'nmae': nmae, 
                'r2': r2, 'mape': mape, 'correlation': correlation}

    def plot_hybrid_reconstruction_detailed(self, reconstruction_results: dict, save_dir: str):
        """
        📊 COMPREHENSIVE HYBRID RECONSTRUCTION VISUALIZATION
        
        Creates multiple plots showing:
        1. Per-band reconstructions
        2. iDWT combination process
        3. Trend comparison
        4. Boundary matching
        5. Full reconstruction vs actual
        """
        
        if 'simple_event' not in reconstruction_results or reconstruction_results['simple_event'] is None:
            print("  ⚠️ No hybrid reconstruction available")
            return
        
        actual = reconstruction_results['actual']
        reconstructed = reconstruction_results['simple_event']
        n_samples = len(actual)
        
        # Check if we have per-band details
        band_reconstructions = reconstruction_results.get('band_reconstructions', {})
        has_bands = len(band_reconstructions) > 0
        
        print(f"\n{'='*80}")
        print("CREATING HYBRID RECONSTRUCTION VISUALIZATIONS")
        print(f"{'='*80}\n")
        
        # ========== PLOT 1: PER-BAND RECONSTRUCTIONS ==========
        if has_bands:
            self._plot_per_band_reconstructions(band_reconstructions, actual, reconstructed, save_dir)
        
        # ========== PLOT 2: RECONSTRUCTION BREAKDOWN ==========
        self._plot_reconstruction_breakdown(actual, reconstructed, save_dir)
        
        # ========== PLOT 3: TREND COMPARISON ==========
        self._plot_trend_comparison(actual, reconstructed, save_dir)
        
        # ========== PLOT 4: BOUNDARY AND TRANSITIONS ==========
        self._plot_boundary_matching(actual, reconstructed, save_dir)
        
        # ========== PLOT 5: ERROR ANALYSIS ==========
        self._plot_error_analysis(actual, reconstructed, save_dir)
        
        # ========== PLOT 6: FREQUENCY SPECTRUM COMPARISON ==========
        self._plot_frequency_spectrum(actual, reconstructed, save_dir)
        
        print(f"✓ Created 6 detailed hybrid reconstruction visualizations\n")
    
    
    def _plot_per_band_reconstructions(self, band_reconstructions: dict, actual: np.ndarray, 
                                        final_reconstructed: np.ndarray, save_dir: str):
        """Plot each frequency band's reconstruction separately"""
        
        n_bands = len(band_reconstructions)
        
        if n_bands == 0:
            return
        
        fig, axes = plt.subplots(n_bands + 2, 1, figsize=(16, 3 * (n_bands + 2)), sharex=True)
        
        # Plot original signal
        axes[0].plot(actual, color='black', linewidth=1.5, label='Actual Signal', alpha=0.8)
        axes[0].set_ylabel('Power (MW)', fontsize=10)
        axes[0].set_title('Original Signal', fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot each band reconstruction
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, (band_name, band_signal) in enumerate(band_reconstructions.items()):
            ax = axes[idx + 1]
            
            # Ensure same length
            if len(band_signal) != len(actual):
                # Interpolate to match length
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, len(band_signal))
                x_new = np.linspace(0, 1, len(actual))
                interpolator = interp1d(x_old, band_signal, kind='linear', fill_value='extrapolate')
                band_signal = interpolator(x_new)
            
            ax.plot(band_signal, color=colors[idx % len(colors)], linewidth=1.0, 
                   label=f'{band_name} Reconstruction', alpha=0.8)
            ax.set_ylabel('Amplitude', fontsize=9)
            ax.set_title(f'Band: {band_name}', fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"Mean: {band_signal.mean():.3f}\nStd: {band_signal.std():.3f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot final reconstruction
        axes[-1].plot(actual, color='black', linewidth=1.5, label='Actual', alpha=0.6)
        axes[-1].plot(final_reconstructed, color='red', linewidth=1.2, 
                     label='Final Reconstruction (iDWT)', alpha=0.8, linestyle='--')
        axes[-1].set_ylabel('Power (MW)', fontsize=10)
        axes[-1].set_xlabel('Time (samples)', fontsize=10)
        axes[-1].set_title('Final Reconstruction (After iDWT)', fontsize=12, fontweight='bold')
        axes[-1].legend(loc='upper right')
        axes[-1].grid(True, alpha=0.3)
        
        # Add metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(actual, final_reconstructed))
        mae = mean_absolute_error(actual, final_reconstructed)
        r2 = r2_score(actual, final_reconstructed)
        
        metrics_text = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.4f}"
        axes[-1].text(0.98, 0.98, metrics_text, transform=axes[-1].transAxes,
                     fontsize=10, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/hybrid_per_band_reconstructions.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"  ✓ Saved: hybrid_per_band_reconstructions.png")
    
    
    def _plot_reconstruction_breakdown(self, actual: np.ndarray, reconstructed: np.ndarray, save_dir: str):
        """Show reconstruction components: trend + refinements"""
        
        from scipy.ndimage import gaussian_filter1d
        
        # Extract trend from both
        trend_window = 50
        actual_trend = gaussian_filter1d(actual, sigma=trend_window)
        recon_trend = gaussian_filter1d(reconstructed, sigma=trend_window)
        
        # Calculate residuals (high-frequency components)
        actual_residual = actual - actual_trend
        recon_residual = reconstructed - recon_trend
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        
        # 1. Trends
        axes[0].plot(actual_trend, color='black', linewidth=2, label='Actual Trend', alpha=0.7)
        axes[0].plot(recon_trend, color='blue', linewidth=2, label='Reconstructed Trend', 
                    alpha=0.7, linestyle='--')
        axes[0].set_ylabel('Power (MW)', fontsize=11)
        axes[0].set_title('Trend Comparison (Smoothed)', fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Trend correlation
        trend_corr = np.corrcoef(actual_trend, recon_trend)[0, 1]
        axes[0].text(0.02, 0.98, f'Correlation: {trend_corr:.4f}', 
                    transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 2. Residuals (Events/Spikes)
        axes[1].plot(actual_residual, color='black', linewidth=0.8, label='Actual Residual', alpha=0.6)
        axes[1].plot(recon_residual, color='red', linewidth=0.8, label='Reconstructed Residual', 
                    alpha=0.6, linestyle='--')
        axes[1].set_ylabel('Residual (MW)', fontsize=11)
        axes[1].set_title('Residuals (Events/Spikes)', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='gray', linestyle=':', linewidth=1)
        
        # 3. Full signals
        axes[2].plot(actual, color='black', linewidth=1.5, label='Actual Signal', alpha=0.7)
        axes[2].plot(reconstructed, color='green', linewidth=1.2, label='Reconstructed Signal', 
                    alpha=0.7, linestyle='--')
        axes[2].set_ylabel('Power (MW)', fontsize=11)
        axes[2].set_title('Full Signal Comparison', fontsize=12, fontweight='bold')
        axes[2].legend(loc='upper right', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        # Add metrics
        from sklearn.metrics import mean_squared_error, r2_score
        rmse = np.sqrt(mean_squared_error(actual, reconstructed))
        r2 = r2_score(actual, reconstructed)
        
        axes[2].text(0.98, 0.98, f'RMSE: {rmse:.2f}\nR²: {r2:.4f}', 
                    transform=axes[2].transAxes, fontsize=10, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 4. Absolute error
        abs_error = np.abs(actual - reconstructed)
        axes[3].fill_between(range(len(abs_error)), abs_error, alpha=0.5, color='red', label='Absolute Error')
        axes[3].plot(abs_error, color='darkred', linewidth=1, alpha=0.8)
        axes[3].set_ylabel('Error (MW)', fontsize=11)
        axes[3].set_xlabel('Time (samples)', fontsize=11)
        axes[3].set_title('Absolute Reconstruction Error', fontsize=12, fontweight='bold')
        axes[3].legend(loc='upper right', fontsize=10)
        axes[3].grid(True, alpha=0.3)
        
        # Error statistics
        mean_error = abs_error.mean()
        max_error = abs_error.max()
        
        axes[3].text(0.98, 0.98, f'Mean Error: {mean_error:.2f}\nMax Error: {max_error:.2f}', 
                    transform=axes[3].transAxes, fontsize=10, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/hybrid_reconstruction_breakdown.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: hybrid_reconstruction_breakdown.png")
    
    
    def _plot_trend_comparison(self, actual: np.ndarray, reconstructed: np.ndarray, save_dir: str):
        """Detailed trend analysis at multiple time scales"""
        
        from scipy.ndimage import gaussian_filter1d
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # Different smoothing windows for multi-scale trend analysis
        windows = [10, 30, 100]  # Short, medium, long-term trends
        
        for idx, window in enumerate(windows):
            # Extract trends
            actual_trend = gaussian_filter1d(actual, sigma=window)
            recon_trend = gaussian_filter1d(reconstructed, sigma=window)
            
            # Left column: Trend comparison
            ax_left = axes[idx, 0]
            ax_left.plot(actual_trend, color='black', linewidth=2, label='Actual', alpha=0.7)
            ax_left.plot(recon_trend, color='blue', linewidth=2, label='Reconstructed', 
                        alpha=0.7, linestyle='--')
            ax_left.set_ylabel('Power (MW)', fontsize=10)
            ax_left.set_title(f'Trend (σ={window})', fontsize=11, fontweight='bold')
            ax_left.legend(loc='upper right')
            ax_left.grid(True, alpha=0.3)
            
            # Correlation
            corr = np.corrcoef(actual_trend, recon_trend)[0, 1]
            ax_left.text(0.02, 0.98, f'Corr: {corr:.4f}', 
                        transform=ax_left.transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
            
            # Right column: Scatter plot
            ax_right = axes[idx, 1]
            ax_right.scatter(actual_trend, recon_trend, alpha=0.3, s=10, color='blue')
            
            # Perfect prediction line
            min_val = min(actual_trend.min(), recon_trend.min())
            max_val = max(actual_trend.max(), recon_trend.max())
            ax_right.plot([min_val, max_val], [min_val, max_val], 
                         'r--', linewidth=2, label='Perfect Prediction')
            
            ax_right.set_xlabel('Actual (MW)', fontsize=10)
            ax_right.set_ylabel('Reconstructed (MW)', fontsize=10)
            ax_right.set_title(f'Scatter (σ={window})', fontsize=11, fontweight='bold')
            ax_right.legend(loc='lower right')
            ax_right.grid(True, alpha=0.3)
            
            # R² score
            from sklearn.metrics import r2_score
            r2 = r2_score(actual_trend, recon_trend)
            ax_right.text(0.02, 0.98, f'R²: {r2:.4f}', 
                         transform=ax_right.transAxes, fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/hybrid_trend_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"  ✓ Saved: hybrid_trend_comparison.png")
    
    
    def _plot_boundary_matching(self, actual: np.ndarray, reconstructed: np.ndarray, save_dir: str):
        """Detailed view of start/end boundary matching"""
        
        n_samples = len(actual)
        boundary_window = min(200, n_samples // 10)  # Show first/last 10%
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # ========== TOP LEFT: START BOUNDARY ==========
        ax = axes[0, 0]
        x_start = range(boundary_window)
        
        ax.plot(x_start, actual[:boundary_window], color='black', linewidth=2, 
               label='Actual', marker='o', markersize=3, alpha=0.7)
        ax.plot(x_start, reconstructed[:boundary_window], color='blue', linewidth=2, 
               label='Reconstructed', marker='s', markersize=3, alpha=0.7, linestyle='--')
        
        # Highlight exact start point
        ax.scatter([0], [actual[0]], color='red', s=100, zorder=5, label='Start Point')
        ax.scatter([0], [reconstructed[0]], color='green', s=80, zorder=5, marker='x')
        
        ax.set_xlabel('Time (samples)', fontsize=11)
        ax.set_ylabel('Power (MW)', fontsize=11)
        ax.set_title('Start Boundary Matching', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Match statistics
        start_error = abs(actual[0] - reconstructed[0])
        start_10_mse = mean_squared_error(actual[:10], reconstructed[:10])
        
        ax.text(0.02, 0.98, 
               f'Start Point Error: {start_error:.4f}\n'
               f'First 10 samples MSE: {start_10_mse:.4f}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # ========== TOP RIGHT: END BOUNDARY ==========
        ax = axes[0, 1]
        x_end = range(n_samples - boundary_window, n_samples)
        
        ax.plot(x_end, actual[-boundary_window:], color='black', linewidth=2, 
               label='Actual', marker='o', markersize=3, alpha=0.7)
        ax.plot(x_end, reconstructed[-boundary_window:], color='blue', linewidth=2, 
               label='Reconstructed', marker='s', markersize=3, alpha=0.7, linestyle='--')
        
        # Highlight exact end point
        ax.scatter([n_samples-1], [actual[-1]], color='red', s=100, zorder=5, label='End Point')
        ax.scatter([n_samples-1], [reconstructed[-1]], color='green', s=80, zorder=5, marker='x')
        
        ax.set_xlabel('Time (samples)', fontsize=11)
        ax.set_ylabel('Power (MW)', fontsize=11)
        ax.set_title('End Boundary Matching', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Match statistics
        end_error = abs(actual[-1] - reconstructed[-1])
        end_10_mse = mean_squared_error(actual[-10:], reconstructed[-10:])
        
        ax.text(0.02, 0.98, 
               f'End Point Error: {end_error:.4f}\n'
               f'Last 10 samples MSE: {end_10_mse:.4f}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # ========== BOTTOM LEFT: BOUNDARY ERROR PROFILE ==========
        ax = axes[1, 0]
        
        error = np.abs(actual - reconstructed)
        
        # Highlight boundary regions
        ax.fill_between(range(boundary_window), error[:boundary_window], 
                        alpha=0.3, color='green', label='Start Region')
        ax.fill_between(range(n_samples - boundary_window, n_samples), 
                        error[-boundary_window:], 
                        alpha=0.3, color='red', label='End Region')
        
        ax.plot(error, color='blue', linewidth=1, alpha=0.7, label='Full Error')
        
        ax.set_xlabel('Time (samples)', fontsize=11)
        ax.set_ylabel('Absolute Error (MW)', fontsize=11)
        ax.set_title('Error Profile with Boundary Regions', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Error statistics
        boundary_error_mean = (error[:boundary_window].mean() + error[-boundary_window:].mean()) / 2
        middle_error_mean = error[boundary_window:-boundary_window].mean()
        
        ax.text(0.98, 0.98, 
               f'Boundary Avg Error: {boundary_error_mean:.2f}\n'
               f'Middle Avg Error: {middle_error_mean:.2f}',
               transform=ax.transAxes, fontsize=9, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # ========== BOTTOM RIGHT: TRANSITION SMOOTHNESS ==========
        ax = axes[1, 1]
        
        # Calculate gradient (rate of change)
        actual_gradient = np.gradient(actual)
        recon_gradient = np.gradient(reconstructed)
        
        ax.plot(actual_gradient, color='black', linewidth=1, label='Actual Gradient', alpha=0.6)
        ax.plot(recon_gradient, color='blue', linewidth=1, label='Reconstructed Gradient', 
               alpha=0.6, linestyle='--')
        
        # Highlight boundary regions
        ax.axvspan(0, boundary_window, alpha=0.2, color='green', label='Start Region')
        ax.axvspan(n_samples - boundary_window, n_samples, alpha=0.2, color='red', label='End Region')
        
        ax.set_xlabel('Time (samples)', fontsize=11)
        ax.set_ylabel('Gradient (MW/sample)', fontsize=11)
        ax.set_title('Transition Smoothness Analysis', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
        
        # Gradient correlation
        grad_corr = np.corrcoef(actual_gradient, recon_gradient)[0, 1]
        ax.text(0.02, 0.98, f'Gradient Correlation: {grad_corr:.4f}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/hybrid_boundary_matching.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: hybrid_boundary_matching.png")
    
    
    def _plot_error_analysis(self, actual: np.ndarray, reconstructed: np.ndarray, save_dir: str):
        """Comprehensive error analysis"""
        
        error = actual - reconstructed
        abs_error = np.abs(error)
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # ========== ERROR TIME SERIES ==========
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(error, color='red', linewidth=0.8, alpha=0.7, label='Error')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax1.fill_between(range(len(error)), error, alpha=0.3, color='red')
        ax1.set_ylabel('Error (MW)', fontsize=11)
        ax1.set_title('Reconstruction Error Over Time', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Statistics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(actual, reconstructed))
        mae = mean_absolute_error(actual, reconstructed)
        mean_error = error.mean()
        std_error = error.std()
        
        ax1.text(0.02, 0.98, 
                f'RMSE: {rmse:.2f}\n'
                f'MAE: {mae:.2f}\n'
                f'Mean Error: {mean_error:.2f}\n'
                f'Std Error: {std_error:.2f}',
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # ========== ERROR HISTOGRAM ==========
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(error, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Error (MW)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Error Distribution', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # ========== ERROR Q-Q PLOT ==========
        ax3 = fig.add_subplot(gs[1, 1])
        from scipy import stats
        stats.probplot(error, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normality Check)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # ========== ERROR AUTOCORRELATION ==========
        ax4 = fig.add_subplot(gs[1, 2])
        
        # Calculate autocorrelation
        max_lag = min(100, len(error) // 4)
        autocorr = [np.corrcoef(error[:-i], error[i:])[0, 1] if i > 0 else 1.0 
                    for i in range(max_lag)]
        
        ax4.bar(range(max_lag), autocorr, color='purple', alpha=0.7, width=1.0)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.axhline(y=0.2, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax4.axhline(y=-0.2, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Lag', fontsize=10)
        ax4.set_ylabel('Autocorrelation', fontsize=10)
        ax4.set_title('Error Autocorrelation', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # ========== ABSOLUTE ERROR OVER TIME ==========
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(abs_error, color='orange', linewidth=0.8, alpha=0.7)
        ax5.fill_between(range(len(abs_error)), abs_error, alpha=0.3, color='orange')
        ax5.set_xlabel('Time (samples)', fontsize=10)
        ax5.set_ylabel('Absolute Error (MW)', fontsize=10)
        ax5.set_title('Absolute Error', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # ========== PERCENTAGE ERROR ==========
        ax6 = fig.add_subplot(gs[2, 1])
        percentage_error = (abs_error / (actual + 1e-8)) * 100
        percentage_error = np.clip(percentage_error, 0, 100)  # Cap at 100%
        
        ax6.plot(percentage_error, color='green', linewidth=0.8, alpha=0.7)
        ax6.set_xlabel('Time (samples)', fontsize=10)
        ax6.set_ylabel('Percentage Error (%)', fontsize=10)
        ax6.set_title('Percentage Error', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        mape = percentage_error.mean()
        ax6.text(0.98, 0.98, f'MAPE: {mape:.2f}%',
                transform=ax6.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # ========== ERROR VS MAGNITUDE ==========
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.scatter(actual, abs_error, alpha=0.3, s=10, color='red')
        ax7.set_xlabel('Actual Power (MW)', fontsize=10)
        ax7.set_ylabel('Absolute Error (MW)', fontsize=10)
        ax7.set_title('Error vs Magnitude', fontsize=11, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # Trend line
        z = np.polyfit(actual, abs_error, 1)
        p = np.poly1d(z)
        ax7.plot(actual, p(actual), "b--", linewidth=2, alpha=0.8, label='Trend')
        ax7.legend(loc='upper left', fontsize=8)
        
        plt.savefig(f"{save_dir}/hybrid_error_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"  ✓ Saved: hybrid_error_analysis.png")
    
    
    def _plot_frequency_spectrum(self, actual: np.ndarray, reconstructed: np.ndarray, save_dir: str):
        """Compare frequency spectra of actual vs reconstructed"""
        
        from scipy.fft import fft, fftfreq
        
        # Calculate FFT
        n = len(actual)
        actual_fft = fft(actual)
        recon_fft = fft(reconstructed)
        
        # Frequencies
        freq = fftfreq(n)
        
        # Only positive frequencies
        pos_mask = freq > 0
        freq_pos = freq[pos_mask]
        actual_power = np.abs(actual_fft[pos_mask]) ** 2
        recon_power = np.abs(recon_fft[pos_mask]) ** 2
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # ========== FULL SPECTRUM ==========
        ax = axes[0, 0]
        ax.semilogy(freq_pos, actual_power, color='black', linewidth=1.5, 
                   label='Actual', alpha=0.7)
        ax.semilogy(freq_pos, recon_power, color='blue', linewidth=1.5, 
                   label='Reconstructed', alpha=0.7, linestyle='--')
        ax.set_xlabel('Frequency', fontsize=11)
        ax.set_ylabel('Power', fontsize=11)
        ax.set_title('Full Frequency Spectrum', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, which='both')
        
        # ========== LOW FREQUENCY ZOOM ==========
        ax = axes[0, 1]
        low_freq_cutoff = 0.1
        low_mask = freq_pos < low_freq_cutoff
        
        ax.semilogy(freq_pos[low_mask], actual_power[low_mask], color='black', 
                   linewidth=2, label='Actual', alpha=0.7)
        ax.semilogy(freq_pos[low_mask], recon_power[low_mask], color='blue', 
                   linewidth=2, label='Reconstructed', alpha=0.7, linestyle='--')
        ax.set_xlabel('Frequency', fontsize=11)
        ax.set_ylabel('Power', fontsize=11)
        ax.set_title('Low Frequency Zoom', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, which='both')
        
        # ========== SPECTRAL DIFFERENCE ==========
        ax = axes[1, 0]
        spectral_diff = np.abs(actual_power - recon_power)
        
        ax.semilogy(freq_pos, spectral_diff, color='red', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Frequency', fontsize=11)
        ax.set_ylabel('Power Difference', fontsize=11)
        ax.set_title('Spectral Difference', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        # ========== SPECTRAL COHERENCE ==========
        ax = axes[1, 1]
        
        # Spectral coherence (correlation in frequency domain)
        coherence = np.minimum(actual_power, recon_power) / np.maximum(actual_power, recon_power + 1e-10)
        
        ax.plot(freq_pos, coherence, color='green', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='0.8 threshold')
        ax.set_xlabel('Frequency', fontsize=11)
        ax.set_ylabel('Coherence', fontsize=11)
        ax.set_title('Spectral Coherence', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        # Average coherence
        avg_coherence = coherence.mean()
        ax.text(0.02, 0.98, f'Avg Coherence: {avg_coherence:.4f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/hybrid_frequency_spectrum.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: hybrid_frequency_spectrum.png")

    def plot_reconstruction_with_events_overlay(self, reconstruction_results: dict, 
                                           results: dict, 
                                           original_signal: np.ndarray,
                                           events_df: pd.DataFrame,
                                           save_dir: str):
        """
        📊 MAIN COMPARISON PLOT
        
        Creates comprehensive visualization showing:
        1. Original time-series vs Reconstructed
        2. Predicted events overlaid on reconstruction
        3. Ground truth events overlaid on original
        4. Side-by-side comparison
        """
        
        actual = reconstruction_results['actual']
        reconstructed = reconstruction_results['simple_event']
        n_samples = len(actual)
        
        h = max(results.keys())  # Use longest horizon
        
        print(f"\n{'='*80}")
        print("CREATING TIME-SERIES + EVENTS OVERLAY VISUALIZATIONS")
        print(f"{'='*80}\n")
        
        # ========== PLOT 1: MAIN COMPARISON WITH EVENTS ==========
        self._plot_main_comparison_with_events(
            actual, reconstructed, results, h, events_df, n_samples, save_dir
        )
        
        # ========== PLOT 2: ZOOMED SECTIONS WITH EVENTS ==========
        self._plot_zoomed_sections_with_events(
            actual, reconstructed, results, h, events_df, n_samples, save_dir
        )
        
        # ========== PLOT 3: EVENT-BY-EVENT COMPARISON ==========
        self._plot_event_by_event_comparison(
            actual, reconstructed, results, h, events_df, n_samples, save_dir
        )
        
        print(f"✓ Created 3 time-series + events overlay visualizations\n")
    
    
    def _plot_main_comparison_with_events(self, actual, reconstructed, results, h, 
                                           events_df, n_samples, save_dir):
        """Main time-series comparison with event overlays"""
        
        fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)
        
        # Get predictions
        pred_occurs = results[h]['predictions']['occurs'].flatten()
        pred_timing = results[h]['predictions']['timing'].flatten()
        
        # Aggregate magnitude/duration if frequency-aware
        if 'magnitude_per_band' in results[h]['predictions']:
            band_weights = {'approximation': 0.5, 'details_4': 0.8, 'details_3': 1.0,
                           'details_2': 0.8, 'details_1': 0.5}
            total_weight = sum(band_weights.values())
            
            pred_magnitude = np.zeros_like(results[h]['predictions']['magnitude_per_band']['approximation'])
            pred_duration = np.zeros_like(results[h]['predictions']['duration_per_band']['approximation'])
            
            for band, weight in band_weights.items():
                pred_magnitude += (weight / total_weight) * results[h]['predictions']['magnitude_per_band'][band]
                pred_duration += (weight / total_weight) * results[h]['predictions']['duration_per_band'][band]
        else:
            pred_magnitude = results[h]['predictions']['magnitude']
            pred_duration = results[h]['predictions']['duration']
        
        pred_magnitude = pred_magnitude.flatten()
        pred_duration = pred_duration.flatten()
        
        # Convert to probabilities
        if pred_occurs.max() > 1.0 or pred_occurs.min() < 0.0:
            pred_occurs_prob = 1.0 / (1.0 + np.exp(-pred_occurs))
        else:
            pred_occurs_prob = pred_occurs
        
        # ========== PANEL 1: ORIGINAL + GROUND TRUTH EVENTS ==========
        ax1 = axes[0]
        ax1.plot(actual, color='black', linewidth=1.5, label='Original Signal', alpha=0.8, zorder=1)
        
        # Overlay ground truth events
        if events_df is not None and not events_df.empty:
            gt_events = events_df[events_df['t1'] < n_samples].copy()
            
            for idx, event in gt_events.iterrows():
                start = int(event['t1'])
                end = int(min(event['t2'], n_samples))
                
                if end <= start or start >= n_samples:
                    continue
                
                # Get magnitude
                if event['event_type'] == 'significant' and pd.notna(event.get('∆w_m')):
                    magnitude = abs(event['∆w_m'])
                elif event['event_type'] == 'stationary' and pd.notna(event.get('σ_s')):
                    magnitude = event['σ_s']
                else:
                    magnitude = 0
                
                # Color by type
                if event['event_type'] == 'significant':
                    color = 'red'
                    label = 'GT: Significant'
                elif event['event_type'] == 'stationary':
                    color = 'blue'
                    label = 'GT: Stationary'
                else:
                    color = 'purple'
                    label = 'GT: Both'
                
                # Highlight event region
                ax1.axvspan(start, end, alpha=0.2, color=color, zorder=0)
                
                # Mark event with vertical line at start
                ax1.axvline(x=start, color=color, linestyle='--', linewidth=1.5, 
                           alpha=0.6, zorder=2)
        
        ax1.set_ylabel('Power (MW)', fontsize=12, fontweight='bold')
        ax1.set_title('Original Signal + Ground Truth Events', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Legend with unique labels
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
        
        # Stats box
        if events_df is not None and not events_df.empty:
            n_gt_events = len(gt_events)
            n_significant = len(gt_events[gt_events['event_type'] == 'significant'])
            n_stationary = len(gt_events[gt_events['event_type'] == 'stationary'])
            
            ax1.text(0.02, 0.98, 
                    f'Ground Truth Events:\n'
                    f'  Total: {n_gt_events}\n'
                    f'  Significant: {n_significant}\n'
                    f'  Stationary: {n_stationary}',
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # ========== PANEL 2: RECONSTRUCTED + PREDICTED EVENTS ==========
        ax2 = axes[1]
        ax2.plot(reconstructed, color='green', linewidth=1.5, label='Reconstructed Signal', 
                alpha=0.8, zorder=1)
        
        # Overlay predicted events
        predicted_event_indices = np.where(pred_occurs_prob[:n_samples] > 0.5)[0]
        
        n_predicted = 0
        for idx in predicted_event_indices:
            if idx >= n_samples:
                break
            
            confidence = pred_occurs_prob[idx]
            magnitude = pred_magnitude[idx]
            duration = int(max(1, pred_duration[idx]))
            
            start = idx
            end = min(idx + duration, n_samples)
            
            if end <= start:
                continue
            
            n_predicted += 1
            
            # Color by confidence
            if confidence > 0.8:
                color = 'darkgreen'
                alpha = 0.3
            elif confidence > 0.6:
                color = 'orange'
                alpha = 0.25
            else:
                color = 'yellow'
                alpha = 0.2
            
            # Highlight event region
            ax2.axvspan(start, end, alpha=alpha, color=color, zorder=0)
            
            # Mark event start
            ax2.axvline(x=start, color=color, linestyle='--', linewidth=1.5, 
                       alpha=0.7, zorder=2)
        
        ax2.set_ylabel('Power (MW)', fontsize=12, fontweight='bold')
        ax2.set_title('Reconstructed Signal + Predicted Events', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=10)
        
        # Stats box
        ax2.text(0.02, 0.98, 
                f'Predicted Events:\n'
                f'  Total: {n_predicted}\n'
                f'  High Conf (>0.8): {np.sum(pred_occurs_prob[predicted_event_indices] > 0.8)}\n'
                f'  Med Conf (0.6-0.8): {np.sum((pred_occurs_prob[predicted_event_indices] > 0.6) & (pred_occurs_prob[predicted_event_indices] <= 0.8))}',
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # ========== PANEL 3: OVERLAY COMPARISON ==========
        ax3 = axes[2]
        ax3.plot(actual, color='black', linewidth=1.5, label='Original', alpha=0.6, zorder=1)
        ax3.plot(reconstructed, color='blue', linewidth=1.2, label='Reconstructed', 
                alpha=0.7, linestyle='--', zorder=2)
        
        # Shade difference
        difference = actual - reconstructed
        ax3.fill_between(range(len(difference)), difference, alpha=0.3, color='red', 
                         label='Difference', zorder=0)
        
        ax3.set_xlabel('Time (samples)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Power (MW)', fontsize=12, fontweight='bold')
        ax3.set_title('Direct Comparison: Original vs Reconstructed', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=10)
        
        # Metrics box
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(actual, reconstructed))
        mae = mean_absolute_error(actual, reconstructed)
        r2 = r2_score(actual, reconstructed)
        
        ax3.text(0.98, 0.98, 
                f'Reconstruction Metrics:\n'
                f'  RMSE: {rmse:.2f}\n'
                f'  MAE: {mae:.2f}\n'
                f'  R²: {r2:.4f}',
                transform=ax3.transAxes, fontsize=10, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/reconstruction_with_events_overlay.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: reconstruction_with_events_overlay.png")
    
    
    def _plot_zoomed_sections_with_events(self, actual, reconstructed, results, h, 
                                           events_df, n_samples, save_dir):
        """Zoomed view of 4 different time sections showing events in detail"""
        
        # Select 4 interesting sections (with events)
        section_length = min(500, n_samples // 8)
        
        # Get predicted event indices
        pred_occurs = results[h]['predictions']['occurs'].flatten()
        if pred_occurs.max() > 1.0 or pred_occurs.min() < 0.0:
            pred_occurs_prob = 1.0 / (1.0 + np.exp(-pred_occurs))
        else:
            pred_occurs_prob = pred_occurs
        
        predicted_event_indices = np.where(pred_occurs_prob[:n_samples] > 0.5)[0]
        
        # Select sections with predicted events
        sections = []
        if len(predicted_event_indices) >= 4:
            # Evenly spaced event locations
            step = len(predicted_event_indices) // 4
            for i in range(4):
                center = predicted_event_indices[i * step]
                start = max(0, center - section_length // 2)
                end = min(n_samples, start + section_length)
                sections.append((start, end))
        else:
            # Fall back to evenly spaced sections
            for i in range(4):
                start = i * (n_samples // 4)
                end = min(start + section_length, n_samples)
                sections.append((start, end))
        
        fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=False)
        
        for idx, (start, end) in enumerate(sections):
            ax = axes[idx]
            
            x_range = range(start, end)
            actual_section = actual[start:end]
            recon_section = reconstructed[start:end]
            
            # Plot signals
            ax.plot(x_range, actual_section, color='black', linewidth=2, 
                   label='Original', alpha=0.8, zorder=3)
            ax.plot(x_range, recon_section, color='blue', linewidth=1.5, 
                   label='Reconstructed', alpha=0.7, linestyle='--', zorder=2)
            
            # Overlay GT events in this section
            if events_df is not None and not events_df.empty:
                section_gt_events = events_df[
                    (events_df['t1'] >= start) & (events_df['t1'] < end)
                ]
                
                for _, event in section_gt_events.iterrows():
                    event_start = int(event['t1'])
                    event_end = int(min(event['t2'], end))
                    
                    if event['event_type'] == 'significant':
                        color = 'red'
                    else:
                        color = 'blue'
                    
                    ax.axvspan(event_start, event_end, alpha=0.15, color=color, zorder=0)
                    ax.axvline(x=event_start, color=color, linestyle=':', 
                              linewidth=2, alpha=0.5, zorder=1, label=f'GT: {event["event_type"]}')
            
            # Overlay predicted events in this section
            section_pred_indices = [i for i in predicted_event_indices if start <= i < end]
            
            for pred_idx in section_pred_indices:
                confidence = pred_occurs_prob[pred_idx]
                
                if confidence > 0.8:
                    color = 'darkgreen'
                else:
                    color = 'orange'
                
                ax.axvline(x=pred_idx, color=color, linestyle='--', 
                          linewidth=2, alpha=0.6, zorder=1)
            
            ax.set_ylabel('Power (MW)', fontsize=11)
            ax.set_title(f'Section {idx+1}: Samples {start}-{end}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Legend with unique labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
            
            # Local metrics
            from sklearn.metrics import mean_squared_error
            local_rmse = np.sqrt(mean_squared_error(actual_section, recon_section))
            local_mae = mean_absolute_error(actual_section, recon_section)
            
            n_gt_local = len(section_gt_events) if events_df is not None and not events_df.empty else 0
            n_pred_local = len(section_pred_indices)
            
            ax.text(0.02, 0.98, 
                   f'Local Metrics:\n'
                   f'  RMSE: {local_rmse:.2f}\n'
                   f'  MAE: {local_mae:.2f}\n'
                   f'GT Events: {n_gt_local}\n'
                   f'Pred Events: {n_pred_local}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        axes[-1].set_xlabel('Time (samples)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/reconstruction_zoomed_sections.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: reconstruction_zoomed_sections.png")
    
    
    def _plot_event_by_event_comparison(self, actual, reconstructed, results, h, 
                                         events_df, n_samples, save_dir):
        """Detailed comparison for individual predicted events"""
        
        pred_occurs = results[h]['predictions']['occurs'].flatten()
        pred_timing = results[h]['predictions']['timing'].flatten()
        
        # Get magnitude/duration
        if 'magnitude_per_band' in results[h]['predictions']:
            band_weights = {'approximation': 0.5, 'details_4': 0.8, 'details_3': 1.0,
                           'details_2': 0.8, 'details_1': 0.5}
            total_weight = sum(band_weights.values())
            
            pred_magnitude = np.zeros_like(results[h]['predictions']['magnitude_per_band']['approximation'])
            pred_duration = np.zeros_like(results[h]['predictions']['duration_per_band']['approximation'])
            
            for band, weight in band_weights.items():
                pred_magnitude += (weight / total_weight) * results[h]['predictions']['magnitude_per_band'][band]
                pred_duration += (weight / total_weight) * results[h]['predictions']['duration_per_band'][band]
        else:
            pred_magnitude = results[h]['predictions']['magnitude']
            pred_duration = results[h]['predictions']['duration']
        
        pred_magnitude = pred_magnitude.flatten()
        pred_duration = pred_duration.flatten()
        
        # Convert to probabilities
        if pred_occurs.max() > 1.0 or pred_occurs.min() < 0.0:
            pred_occurs_prob = 1.0 / (1.0 + np.exp(-pred_occurs))
        else:
            pred_occurs_prob = pred_occurs
        
        # Get top 12 predicted events (by confidence)
        predicted_event_indices = np.where(pred_occurs_prob[:n_samples] > 0.5)[0]
        
        if len(predicted_event_indices) == 0:
            print(f"  ⚠️ No predicted events found, skipping event-by-event comparison")
            return
        
        # Sort by confidence
        event_confidences = pred_occurs_prob[predicted_event_indices]
        top_event_indices = predicted_event_indices[np.argsort(event_confidences)[-12:][::-1]]
        
        n_events = min(12, len(top_event_indices))
        
        fig, axes = plt.subplots(4, 3, figsize=(20, 16))
        axes = axes.flatten()
        
        for plot_idx in range(12):
            ax = axes[plot_idx]
            
            if plot_idx >= n_events:
                ax.axis('off')
                continue
            
            event_idx = top_event_indices[plot_idx]
            confidence = pred_occurs_prob[event_idx]
            magnitude = pred_magnitude[event_idx]
            duration = int(pred_duration[event_idx])
            
            # Define window around event
            window_size = max(100, duration * 3)
            start = max(0, event_idx - window_size // 2)
            end = min(n_samples, event_idx + window_size // 2)
            
            x_range = range(start, end)
            actual_window = actual[start:end]
            recon_window = reconstructed[start:end]
            
            # Plot signals
            ax.plot(x_range, actual_window, color='black', linewidth=1.5, 
                   label='Original', alpha=0.8)
            ax.plot(x_range, recon_window, color='blue', linewidth=1.2, 
                   label='Reconstructed', alpha=0.7, linestyle='--')
            
            # Highlight predicted event
            event_end = min(event_idx + duration, end)
            ax.axvspan(event_idx, event_end, alpha=0.3, color='green', 
                      label='Predicted Event', zorder=0)
            ax.axvline(x=event_idx, color='green', linestyle='--', linewidth=2, alpha=0.8)
            
            # Check for GT events in this window
            if events_df is not None and not events_df.empty:
                window_gt_events = events_df[
                    (events_df['t1'] >= start) & (events_df['t1'] < end)
                ]
                
                for _, gt_event in window_gt_events.iterrows():
                    gt_start = int(gt_event['t1'])
                    gt_end = int(min(gt_event['t2'], end))
                    
                    ax.axvspan(gt_start, gt_end, alpha=0.2, color='red', zorder=0)
                    ax.axvline(x=gt_start, color='red', linestyle=':', linewidth=2, alpha=0.6)
            
            ax.set_ylabel('Power (MW)', fontsize=9)
            ax.set_title(f'Event #{plot_idx+1} (Conf: {confidence:.3f})', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=7)
            
            # Event details
            ax.text(0.02, 0.98, 
                   f'Magnitude: {magnitude:.2f}\n'
                   f'Duration: {duration} steps',
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.suptitle('Event-by-Event Comparison (Top 12 Predicted Events by Confidence)', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/reconstruction_event_by_event.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: reconstruction_event_by_event.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: HAWKES PROCESS LAYER (EVENT CAUSALITY MODELING)
# ═══════════════════════════════════════════════════════════════════════════

class HawkesProcessLayer(nn.Module):
    """
    Simple Hawkes Process Layer - Models Temporal Event Causality
    
    Learns how past events influence future event probability:
    λ(t) = μ + Σ α_k * exp(-β_k * (t - t_i))
    
    where:
    - μ = base intensity (background event rate)
    - α = excitation (how much past events increase future probability)
    - β = decay rate (how quickly influence fades)
    - k = kernel index (multiple timescales)
    
    This helps the model understand:
    - Event cascades (one ramp triggers another)
    - Temporal clustering (events happen in bursts)
    - Inter-event dependencies
    
    Uses explicit loops for stability (slower but reliable).
    """
    
    def __init__(self, input_size, hawkes_dim=24, num_kernels=3, dropout=0.45):
        super(HawkesProcessLayer, self).__init__()  
        
        self.hawkes_dim = hawkes_dim
        self.num_kernels = num_kernels
        
        # Base intensity network
        self.base_intensity_net = nn.Sequential(
            nn.Linear(input_size, hawkes_dim),
            nn.Softplus(),
            nn.Dropout(dropout)
        )
        
        # Excitation network
        self.excitation_net = nn.Sequential(
            nn.Linear(input_size, hawkes_dim * num_kernels),
            nn.Softplus(),
            nn.Dropout(dropout)
        )
        
        # Decay network
        self.decay_net = nn.Sequential(
            nn.Linear(input_size, hawkes_dim * num_kernels),
            nn.Softplus(),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hawkes_dim, hawkes_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        print(f"  Hawkes Process Layer initialized (stable loop-based version)")
        print(f"    - Hawkes dimension: {hawkes_dim}")
        print(f"    - Number of kernels: {num_kernels}")
        print(f"    - Dropout: {dropout}")
    
    def forward(self, x):
        """
        Compute Hawkes intensity features from input sequence
        
        Args:
            x: Input features (batch, seq_len, input_size)
        
        Returns:
            intensity_features: Hawkes intensity encoding (batch, seq_len, hawkes_dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Base intensity: (batch, seq_len, hawkes_dim)
        mu = self.base_intensity_net(x)
        
        # Parameters: (batch, seq_len, hawkes_dim, num_kernels)
        alpha = self.excitation_net(x).view(batch_size, seq_len, self.hawkes_dim, self.num_kernels)
        beta = self.decay_net(x).view(batch_size, seq_len, self.hawkes_dim, self.num_kernels)
        
        # Initialize intensity with base rate
        intensity = mu.clone()
        
        # Compute historical influence at each timestep (explicit loop for stability)
        for t in range(1, seq_len):
            # Accumulate influence from all past events
            past_influence = torch.zeros(batch_size, self.hawkes_dim, device=device)
            
            for s in range(t):
                # Time difference
                dt = t - s
                
                # Sum over all kernels (different timescales)
                for k in range(self.num_kernels):
                    # Exponential kernel: exp(-β * Δt)
                    kernel = torch.exp(-beta[:, s, :, k] * dt)
                    
                    # Weighted influence: α * kernel
                    past_influence += alpha[:, s, :, k] * kernel
            
            # Add historical influence to base intensity at time t
            intensity[:, t] = intensity[:, t] + past_influence
        
        # Project to output space
        intensity_features = self.output_proj(intensity)
        
        return intensity_features


class OptimizedHawkesLayer(nn.Module):
    """
    OPTIMIZED Hawkes Process Layer (FIXED - vectorized computation)
    
    Uses vectorized operations for speed.
    More complex but faster than loop-based version.
    """
    
    def __init__(self, input_size, hawkes_dim=24, num_kernels=3, dropout=0.45):
        super(OptimizedHawkesLayer, self).__init__()  
        
        self.hawkes_dim = hawkes_dim
        self.num_kernels = num_kernels
        
        # Parameter networks
        self.base_intensity_net = nn.Sequential(
            nn.Linear(input_size, hawkes_dim),
            nn.Softplus(),
            nn.Dropout(dropout)
        )
        
        self.excitation_net = nn.Sequential(
            nn.Linear(input_size, hawkes_dim * num_kernels),
            nn.Softplus(),
            nn.Dropout(dropout)
        )
        
        self.decay_net = nn.Sequential(
            nn.Linear(input_size, hawkes_dim * num_kernels),
            nn.Softplus(),
            nn.Dropout(dropout)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hawkes_dim, hawkes_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        print(f"  Optimized Hawkes Process Layer (vectorized - faster)")
    
    def forward(self, x):
        """
        Proper dimension handling for vectorized computation
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Base intensity: (batch, seq_len, hawkes_dim)
        mu = self.base_intensity_net(x)
        
        # Parameters: (batch, seq_len, hawkes_dim, num_kernels)
        alpha = self.excitation_net(x).view(batch_size, seq_len, self.hawkes_dim, self.num_kernels)
        beta = self.decay_net(x).view(batch_size, seq_len, self.hawkes_dim, self.num_kernels)
        
        # Initialize intensity with base rate
        intensity = mu.clone()
        
        # Create time difference matrix: (seq_len, seq_len)
        time_indices = torch.arange(seq_len, device=device).float()
        dt_matrix = time_indices.unsqueeze(0) - time_indices.unsqueeze(1)  # (seq_len, seq_len)
        
        # Mask: only past influences future (lower triangular, excluding diagonal)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1)
        dt_matrix = dt_matrix * mask  # Zero out upper triangle
        dt_matrix = torch.clamp(dt_matrix, min=0)  # Ensure non-negative
        
        # Compute influence for each kernel separately
        for k in range(self.num_kernels):
            # Get parameters for this kernel: (batch, seq_len, hawkes_dim)
            alpha_k = alpha[:, :, :, k]  # (batch, seq_len, hawkes_dim)
            beta_k = beta[:, :, :, k]    # (batch, seq_len, hawkes_dim)
            
            # Expand dimensions for broadcasting
            beta_expanded = beta_k.unsqueeze(-1)  # (batch, seq_len, hawkes_dim, 1)
            dt_expanded = dt_matrix.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, seq_len)
            
            # Compute kernels: (batch, seq_len, hawkes_dim, seq_len)
            kernels = torch.exp(-beta_expanded * dt_expanded)
            
            # Weight by excitation: (batch, seq_len, hawkes_dim, seq_len)
            alpha_expanded = alpha_k.unsqueeze(-1)  # (batch, seq_len, hawkes_dim, 1)
            weighted_kernels = alpha_expanded * kernels
            
            # Sum over all past events (dimension -1): (batch, seq_len, hawkes_dim)
            past_influence = weighted_kernels.sum(dim=-1)
            
            # Add to total intensity
            intensity = intensity + past_influence
        
        # Project to output space
        intensity_features = self.output_proj(intensity)
        
        return intensity_features

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: HAWKES + Transformer HYBRID ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

class HawkesTransformerHybrid(nn.Module):
    """
    Hawkes + Transformer Hybrid Architecture
    
    Architecture:
    Input Features → Hawkes Layer → [Features + Hawkes] → Transformer → Per-Band Heads
    
    Improvements over vanilla Transformer:
    - Hawkes layer models event causality (one ramp triggers another)
    - Transformer learns patterns from both raw features AND Hawkes intensity
    - Per-band heads predict magnitude/duration per frequency band
    
    Regularization to prevent overfitting:
    - Small Hawkes dimension (24)
    - Aggressive dropout (0.45)
    - Optional Staged training (train Hawkes first, then fine-tune all)
    """
    
    def __init__(self, input_size, hidden_size, num_layers, dropout, 
                 horizon_hours_list, short_horizon_threshold_timesteps, 
                 frequency_bands=None, use_short_horizon_opt=True, 
                 use_frequency_aware=True,
                 use_hawkes=True, hawkes_dim=24, hawkes_kernels=3, hawkes_dropout=0.45,
                 nhead=8):
        super(HawkesTransformerHybrid, self).__init__()
        
        self.hidden_size = hidden_size
        self.horizons = list(horizon_hours_list)
        self.short_horizon_threshold = short_horizon_threshold_timesteps
        self.use_short_horizon_opt = use_short_horizon_opt
        self.use_frequency_aware = use_frequency_aware
        self.use_hawkes = use_hawkes
        
        # Frequency bands
        if frequency_bands is None:
            frequency_bands = ['approximation', 'details_4', 'details_3', 'details_2', 'details_1']
        self.frequency_bands = frequency_bands
        
        # Safe names for ModuleDict
        def _safe(h):
            if isinstance(h, float):
                return "h" + str(h).replace(".", "p")
            return f"h{h}"
        
        self._name_map = {h: _safe(h) for h in self.horizons}
        
        # HAWKES PROCESS LAYER (optional)
        if use_hawkes:
            print(f"\n Adding Hawkes Process Layer to Architecture:")
            self.hawkes_layer = OptimizedHawkesLayer(
                input_size=input_size,
                hawkes_dim=hawkes_dim,
                num_kernels=hawkes_kernels,
                dropout=hawkes_dropout
            )
            transformer_input_size = input_size + hawkes_dim  # Concatenate Hawkes + original
        else:
            self.hawkes_layer = None
            transformer_input_size = input_size
        
        # Input projection to hidden_size
        self.input_projection = nn.Linear(transformer_input_size, hidden_size)
        
        # Positional encoding for MAIN transformer
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)
        
        # Main Transformer encoder (takes original + Hawkes features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)
        
        # Short-horizon Transformer (optional)
        if use_short_horizon_opt:
            short_hidden = hidden_size // 2
            
            short_encoder_layer = nn.TransformerEncoderLayer(
                d_model=short_hidden,
                nhead=max(1, nhead // 2),
                dim_feedforward=short_hidden * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.short_horizon_projection = nn.Linear(transformer_input_size, short_hidden)
            
            #  SEPARATE positional encoding for short-horizon transformer
            self.short_positional_encoding = PositionalEncoding(short_hidden, dropout)
            
            self.short_horizon_transformer = nn.TransformerEncoder(
                short_encoder_layer,
                num_layers=1
            )
            self.short_dropout = nn.Dropout(dropout)
        
        # ========== PREDICTION HEADS ==========
        self.occurrence_heads = nn.ModuleDict()
        self.type_heads = nn.ModuleDict()
        self.timing_heads = nn.ModuleDict()
        
        if use_frequency_aware:
            self.magnitude_heads_per_band = nn.ModuleDict()
            self.duration_heads_per_band = nn.ModuleDict()
        else:
            self.magnitude_heads = nn.ModuleDict()
            self.duration_heads = nn.ModuleDict()
        
        for h in self.horizons:
            name = self._name_map[h]
            head_hidden = hidden_size // 2 if (h <= 1.0 and use_short_horizon_opt) else hidden_size
            inner = max(1, head_hidden // 2)
            
            # Global heads with GeLU activation
            self.occurrence_heads[name] = nn.Sequential(
                nn.Linear(head_hidden, inner),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(inner, 1)
            )
            
            self.type_heads[name] = nn.Sequential(
                nn.Linear(head_hidden, inner),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(inner, 4)
            )
            
            self.timing_heads[name] = nn.Sequential(
                nn.Linear(head_hidden, inner),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(inner, 1)
            )
            
            # Per-band magnitude and duration heads
            if use_frequency_aware:
                self.magnitude_heads_per_band[name] = nn.ModuleDict()
                self.duration_heads_per_band[name] = nn.ModuleDict()
                
                for band in self.frequency_bands:
                    self.magnitude_heads_per_band[name][band] = nn.Sequential(
                        nn.Linear(head_hidden, inner),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(inner, 1)
                    )
                    
                    self.duration_heads_per_band[name][band] = nn.Sequential(
                        nn.Linear(head_hidden, inner),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(inner, 1)
                    )
            else:
                self.magnitude_heads[name] = nn.Sequential(
                    nn.Linear(head_hidden, inner),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(inner, 1)
                )
                
                self.duration_heads[name] = nn.Sequential(
                    nn.Linear(head_hidden, inner),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(inner, 1)
                )
        
        # Print model summary
        total_params = sum(p.numel() for p in self.parameters())
        if use_hawkes:
            hawkes_params = sum(p.numel() for p in self.hawkes_layer.parameters())
            transformer_params = sum(p.numel() for p in self.transformer.parameters())
            print(f"\n Model Summary:")
            print(f"  Hawkes layer: {hawkes_params:,} parameters")
            print(f"  Transformer: {transformer_params:,} parameters")
            print(f"  Total: {total_params:,} parameters (+{hawkes_params/total_params*100:.1f}% from Hawkes)")
        else:
            print(f"\n Model Summary: {total_params:,} parameters")
    
    def forward(self, x):
        """Forward pass with optional Hawkes intensity features"""
        
        # Get Hawkes intensity features (if enabled)
        if self.use_hawkes:
            hawkes_features = self.hawkes_layer(x)  # (batch, seq, hawkes_dim)
            
            # Concatenate original features with Hawkes intensity
            x_combined = torch.cat([x, hawkes_features], dim=-1)
        else:
            x_combined = x
        
        # Encode sequence with Transformer
        x_proj = self.input_projection(x_combined)
        x_pos = self.positional_encoding(x_proj)
        transformer_out = self.transformer(x_pos)
        last_output_main = self.dropout(transformer_out[:, -1, :])
        
        if self.use_short_horizon_opt:
            x_short_proj = self.short_horizon_projection(x_combined)
            # 👇 Use SEPARATE positional encoding
            x_short_pos = self.short_positional_encoding(x_short_proj)
            short_transformer_out = self.short_horizon_transformer(x_short_pos)
            last_output_short = self.short_dropout(short_transformer_out[:, -1, :])
        else:
            last_output_short = None
        
        # Generate predictions for each horizon
        predictions = {}
        for h in self.horizons:
            name = self._name_map[h]
            use_short = (h <= 1.0 and self.use_short_horizon_opt)
            encoding = last_output_short if use_short else last_output_main
            
            # Global predictions
            predictions[h] = {
                'occurs': self.occurrence_heads[name](encoding),
                'type': self.type_heads[name](encoding),
                'timing': self.timing_heads[name](encoding)
            }
            
            # Frequency-aware magnitude and duration predictions
            if self.use_frequency_aware:
                predictions[h]['magnitude_per_band'] = {}
                predictions[h]['duration_per_band'] = {}
                
                for band in self.frequency_bands:
                    predictions[h]['magnitude_per_band'][band] = \
                        self.magnitude_heads_per_band[name][band](encoding)
                    predictions[h]['duration_per_band'][band] = \
                        self.duration_heads_per_band[name][band](encoding)
            else:
                predictions[h]['magnitude'] = self.magnitude_heads[name](encoding)
                predictions[h]['duration'] = self.duration_heads[name](encoding)
        
        return predictions


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: DATASET AND LOSS (FREQUENCY-AWARE!)
# ═══════════════════════════════════════════════════════════════════════════

class MultiTaskDataset(Dataset):
    """ Dataset with per-band labels for frequency-aware training"""
    
    def __init__(self, features, labels, sequence_length, horizon_hours_list, frequency_bands=None, use_frequency_aware=True):
        self.features = torch.FloatTensor(features)
        self.sequence_length = sequence_length
        self.horizons = horizon_hours_list
        self.use_frequency_aware = use_frequency_aware
        
        if frequency_bands is None:
            frequency_bands = ['approximation', 'details_4', 'details_3', 'details_2', 'details_1']
        self.frequency_bands = frequency_bands
        
        self.labels = {}
        for h in horizon_hours_list:
            self.labels[h] = {
                'occurs': torch.FloatTensor(labels[f'event_occurs_{h}h'].values),
                'type': torch.LongTensor(labels[f'event_type_{h}h'].values),
                'timing': torch.FloatTensor(labels[f'event_timing_{h}h'].values)
            }
            
            # Per-band labels
            if use_frequency_aware:
                self.labels[h]['magnitude_per_band'] = {}
                self.labels[h]['duration_per_band'] = {}
                
                for band in self.frequency_bands:
                    mag_col = f'event_magnitude_{h}h_{band}'
                    dur_col = f'event_duration_{h}h_{band}'
                    
                    if mag_col in labels.columns:
                        self.labels[h]['magnitude_per_band'][band] = torch.FloatTensor(labels[mag_col].values)
                    else:
                        # Fallback if column doesn't exist
                        self.labels[h]['magnitude_per_band'][band] = torch.FloatTensor(labels[f'event_magnitude_{h}h'].values)
                    
                    if dur_col in labels.columns:
                        self.labels[h]['duration_per_band'][band] = torch.FloatTensor(labels[dur_col].values)
                    else:
                        # Fallback if column doesn't exist
                        self.labels[h]['duration_per_band'][band] = torch.FloatTensor(labels[f'event_duration_{h}h'].values)
            else:
                # Original single labels
                self.labels[h]['magnitude'] = torch.FloatTensor(labels[f'event_magnitude_{h}h'].values)
                self.labels[h]['duration'] = torch.FloatTensor(labels[f'event_duration_{h}h'].values)
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        X = self.features[idx:idx + self.sequence_length]
        label_idx = idx + self.sequence_length
        
        y = {}
        for h in self.horizons:
            y[h] = {
                'occurs': self.labels[h]['occurs'][label_idx],
                'type': self.labels[h]['type'][label_idx],
                'timing': self.labels[h]['timing'][label_idx]
            }
            
            if self.use_frequency_aware:
                y[h]['magnitude_per_band'] = {band: self.labels[h]['magnitude_per_band'][band][label_idx] 
                                              for band in self.frequency_bands}
                y[h]['duration_per_band'] = {band: self.labels[h]['duration_per_band'][band][label_idx] 
                                            for band in self.frequency_bands}
            else:
                y[h]['magnitude'] = self.labels[h]['magnitude'][label_idx]
                y[h]['duration'] = self.labels[h]['duration'][label_idx]
        
        return X, y


class BandWeightedLoss(nn.Module):
    """
    Band-Weighted Multi-Task Loss with Hawkes Support
    
    Weights different frequency bands based on their magnitude scales:
    - approximation: Large, slow trends (weight = 2.0)
    - details_4: Tiny, fast fluctuations (weight = 50.0)
    - details_3: Medium ramps (weight = 10.0)
    - details_2/1: Small, fast events (weight = 3.0/1.5)
    
    Uses Huber loss for robustness to outliers.
    """
    
    def __init__(self, horizons, frequency_bands=None, pos_weights=None, 
                 use_frequency_aware=True, huber_delta=1.0):
        super(BandWeightedLoss, self).__init__()
        
        self.horizons = horizons
        self.use_frequency_aware = use_frequency_aware
        self.huber_delta = huber_delta
        
        if frequency_bands is None:
            frequency_bands = ['approximation', 'details_4', 'details_3', 'details_2', 'details_1']
        self.frequency_bands = frequency_bands
        
        # ENHANCED: Higher weight for approximation (was struggling)
        self.band_weights = {
            'approximation': 1.5,   # ✅ Reduced
            'details_4': 4.0,       
            'details_3': 2.5,       # ✅ Reduced:
            'details_2': 0.5,       # ✅ Reduced
            'details_1': 0.5        # ✅ Reduced
        }
        
        # Task weights (same as before)
        self.task_weights = {
            'occurs': 2.0,        # ✅ INCREASE from 1.0 (prioritize detection!)
            'type': 1.2,          # ✅ Increase slightly
            'magnitude': 2.0,     # Keep same
            'duration': 2.0,      # ✅ Reduce
            'timing': 1.0         # Keep same
        }
        
        # Occurrence loss (binary cross-entropy)
        if pos_weights is not None:
            self.bce_losses = {
                h: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weights[h]]))
                for h in horizons
            }
        else:
            self.bce_losses = {h: nn.BCEWithLogitsLoss() for h in horizons}
        
        # Type loss (cross-entropy)
        self.ce_loss = nn.CrossEntropyLoss()
        
        print(f"\n Band-Weighted Loss Configuration:")
        print(f"  Use frequency-aware: {use_frequency_aware}")
        print(f"  Huber delta: {huber_delta}")
        print(f"  Band weights:")
        for band, weight in self.band_weights.items():
            print(f"    {band:15s}: {weight:5.1f}")
        print(f"  Task weights:")
        for task, weight in self.task_weights.items():
            print(f"    {task:15s}: {weight:5.1f}")
    
    def huber_loss(self, pred, target, delta=1.0):
        """Huber loss for robust regression"""
        error = pred - target
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + delta * linear
        return loss.mean()
    
    def forward(self, predictions, targets):
        """Compute weighted multi-task loss"""
        
        total_loss = 0.0
        loss_dict = {}
        
        for h in self.horizons:
            horizon_loss = 0.0
            
            # Occurrence loss
            occurs_loss = self.bce_losses[h](
                predictions[h]['occurs'].squeeze(),
                targets[h]['occurs'].float()
            )
            horizon_loss += self.task_weights['occurs'] * occurs_loss
            loss_dict[f'{h}h_occurs'] = occurs_loss.item()
            
            # Type loss
            type_loss = self.ce_loss(
                predictions[h]['type'],
                targets[h]['type'].long()
            )
            horizon_loss += self.task_weights['type'] * type_loss
            loss_dict[f'{h}h_type'] = type_loss.item()
            
            # Timing loss
            timing_loss = self.huber_loss(
                predictions[h]['timing'].squeeze(),
                targets[h]['timing'].float(),
                delta=self.huber_delta
            )
            horizon_loss += self.task_weights['timing'] * timing_loss
            loss_dict[f'{h}h_timing'] = timing_loss.item()
            
            # Frequency-aware magnitude and duration losses
            if self.use_frequency_aware:
                mag_loss_total = 0.0
                dur_loss_total = 0.0
                
                for band in self.frequency_bands:
                    # Magnitude loss (band-weighted Huber)
                    mag_pred = predictions[h]['magnitude_per_band'][band].squeeze()
                    mag_target = targets[h]['magnitude_per_band'][band].float()
                    
                    mag_loss_band = self.huber_loss(mag_pred, mag_target, delta=self.huber_delta)
                    mag_loss_total += self.band_weights[band] * mag_loss_band
                    
                    # Duration loss (band-weighted Huber)
                    dur_pred = predictions[h]['duration_per_band'][band].squeeze()
                    dur_target = targets[h]['duration_per_band'][band].float()
                    
                    dur_loss_band = self.huber_loss(dur_pred, dur_target, delta=self.huber_delta)
                    dur_loss_total += self.band_weights[band] * dur_loss_band
                
                # Average across bands and add to horizon loss
                horizon_loss += self.task_weights['magnitude'] * (mag_loss_total / len(self.frequency_bands))
                horizon_loss += self.task_weights['duration'] * (dur_loss_total / len(self.frequency_bands))
                
                loss_dict[f'{h}h_magnitude'] = mag_loss_total.item() / len(self.frequency_bands)
                loss_dict[f'{h}h_duration'] = dur_loss_total.item() / len(self.frequency_bands)
            else:
                # Single magnitude/duration loss
                mag_loss = self.huber_loss(
                    predictions[h]['magnitude'].squeeze(),
                    targets[h]['magnitude'].float(),
                    delta=self.huber_delta
                )
                dur_loss = self.huber_loss(
                    predictions[h]['duration'].squeeze(),
                    targets[h]['duration'].float(),
                    delta=self.huber_delta
                )
                
                horizon_loss += self.task_weights['magnitude'] * mag_loss
                horizon_loss += self.task_weights['duration'] * dur_loss
                
                loss_dict[f'{h}h_magnitude'] = mag_loss.item()
                loss_dict[f'{h}h_duration'] = dur_loss.item()
            
            total_loss += horizon_loss
            loss_dict[f'{h}h_total'] = horizon_loss.item()
        
        return total_loss, loss_dict

class MultiHorizonLabeler:
    """Create labels with per-band magnitude and duration (NORMALIZED)"""
    
    def __init__(self, config):
        self.config = config
        self.band_stats = {}  # ✨ Store band statistics for normalization
        
    def create_labels(self, df: pd.DataFrame, events_df: pd.DataFrame, 
                      event_results_per_band: dict = None,
                      dwt_decomposition: dict = None) -> pd.DataFrame:  # ✨ ADD dwt_decomposition
        """Create labels for all horizons (with per-band labels, NORMALIZED)"""
        
        print(f"\n{'='*80}")
        print("CREATING LABELS (FREQUENCY-AWARE, NORMALIZED)")
        print(f"{'='*80}\n")
        
        # ✨ COMPUTE BAND STATISTICS FOR NORMALIZATION
        if self.config.USE_FREQUENCY_AWARE_PREDICTION and dwt_decomposition:
            self._compute_band_statistics(dwt_decomposition)
        
        n_samples = len(df)
        labels = pd.DataFrame(index=df.index)
        
        horizons_timesteps = self.config.prediction_horizons
        horizons_hours = self.config.PREDICTION_HORIZONS_HOURS
        
        # Initialize global labels
        for h_steps, h_hours in zip(horizons_timesteps, horizons_hours):
            labels[f'event_occurs_{h_hours}h'] = 0
            labels[f'event_type_{h_hours}h'] = 0
            labels[f'event_magnitude_{h_hours}h'] = 0.0
            labels[f'event_duration_{h_hours}h'] = 0.0
            labels[f'event_timing_{h_hours}h'] = float(h_hours)
            
            # Initialize per-band labels
            if self.config.USE_FREQUENCY_AWARE_PREDICTION and event_results_per_band:
                for band in self.config.FREQUENCY_BANDS:
                    labels[f'event_magnitude_{h_hours}h_{band}'] = 0.0
                    labels[f'event_duration_{h_hours}h_{band}'] = 0.0
        
        if events_df.empty:
            print("⚠ No events to label\n" + "="*80 + "\n")
            return labels
        
        # Global labels (from combined events)
        labels = self._create_global_labels(labels, events_df, horizons_timesteps, horizons_hours, n_samples)
        
        # Per-band labels (from per-band events)
        if self.config.USE_FREQUENCY_AWARE_PREDICTION and event_results_per_band and 'per_band' in event_results_per_band:
            labels = self._create_per_band_labels(labels, event_results_per_band['per_band'], 
                                                   horizons_timesteps, horizons_hours, n_samples)
        
        print(f"\n{'='*80}\n")
        return labels
    
    def _compute_band_statistics(self, dwt_decomposition: dict):
        """✨ CORRECTED: Compute statistics for each band for normalization"""
        
        print(f"  Computing band statistics for normalization...")
        
        # Debug: Check what keys are actually in the decomposition
        print(f"  Available keys: {list(dwt_decomposition.keys())}")
        
        # Approximation band
        if 'approximation' in dwt_decomposition:
            approx_coeffs = dwt_decomposition['approximation']
            self.band_stats['approximation'] = {
                'std': approx_coeffs.std(),
                'median': np.median(np.abs(approx_coeffs))
            }
            print(f"    approximation: std={self.band_stats['approximation']['std']:.2f}")
        else:
            print(f"    ⚠️ No 'approximation' key found!")
        
        # ✅ CORRECTED: Details bands are stored as separate keys, not in a list
        for level in range(1, self.config.DWT_LEVEL + 1):
            band_name = f'details_{level}'
            
            # Check if this band exists in the decomposition
            if band_name in dwt_decomposition:
                detail_coeffs = dwt_decomposition[band_name]
                
                self.band_stats[band_name] = {
                    'std': detail_coeffs.std(),
                    'median': np.median(np.abs(detail_coeffs))
                }
                print(f"    {band_name}: std={self.band_stats[band_name]['std']:.2f}")
            else:
                print(f"    ⚠️ {band_name} not found in decomposition!")
        
    def _create_global_labels(self, labels: pd.DataFrame, events_df: pd.DataFrame,
                             horizons_timesteps: list, horizons_hours: list, n_samples: int) -> pd.DataFrame:
        """Create global event labels (occurrence, type, timing, aggregate magnitude/duration)"""
        
        print(f"\nCreating GLOBAL labels from {len(events_df)} combined events...")
        
        # Prepare event data
        events_sorted = events_df.sort_values('t1').reset_index(drop=True)
        event_starts = events_sorted['t1'].values.astype(int)
        event_ends = events_sorted['t2'].values.astype(int)
        
        # Extract magnitudes safely
        magnitudes = np.zeros(len(events_sorted))
        for idx, event in events_sorted.iterrows():
            if event['event_type'] == 'significant' and pd.notna(event.get('∆w_m')):
                magnitudes[idx] = abs(event['∆w_m'])
            elif event['event_type'] == 'stationary' and pd.notna(event.get('σ_s')):
                magnitudes[idx] = event['σ_s']
        
        # Extract durations safely
        durations = np.ones(len(events_sorted))
        for idx, event in events_sorted.iterrows():
            if event['event_type'] == 'significant' and pd.notna(event.get('∆t_m')):
                durations[idx] = event['∆t_m']
            elif event['event_type'] == 'stationary' and pd.notna(event.get('∆t_s')):
                durations[idx] = event['∆t_s']
        
        event_types = events_sorted['event_type'].values
        
        for h_steps, h_hours in zip(horizons_timesteps, horizons_hours):
            print(f"  Horizon {h_hours}h ({h_steps} timesteps)...", end=" ", flush=True)
            
            # Initialize arrays
            occurs = np.zeros(n_samples, dtype=np.int8)
            types = np.zeros(n_samples, dtype=np.int8)
            mags = np.zeros(n_samples, dtype=np.float32)
            durs = np.zeros(n_samples, dtype=np.float32)
            timings = np.full(n_samples, float(h_hours), dtype=np.float32)
            
            valid_indices = np.arange(0, n_samples - h_steps)
            
            if len(valid_indices) > 0:
                for idx in valid_indices:
                    future_end = idx + h_steps
                    
                    # Find events using binary search
                    start_idx = np.searchsorted(event_starts, idx, side='left')
                    end_idx = np.searchsorted(event_starts, future_end, side='right')
                    
                    if start_idx < end_idx:
                        occurs[idx] = 1
                        
                        window_mask = (event_starts >= idx) & (event_starts < future_end)
                        window_indices = np.where(window_mask)[0]
                        
                        if len(window_indices) > 0:
                            # Event type
                            window_types = event_types[window_indices]
                            has_sig = np.any(window_types == 'significant')
                            has_stat = np.any(window_types == 'stationary')
                            types[idx] = 3 if (has_sig and has_stat) else (1 if has_sig else 2)
                            
                            # Timing
                            first_event_start = event_starts[window_indices[0]]
                            timing_timesteps = first_event_start - idx
                            timings[idx] = self.config.timesteps_to_hours(max(1, timing_timesteps))
                            
                            # Magnitude (max in window) - KEEP RAW for global labels
                            window_mags = magnitudes[window_indices]
                            mags[idx] = np.max(window_mags)
                            
                            # Duration (mean in window)
                            window_durs = durations[window_indices]
                            durs[idx] = np.mean(window_durs)
            
            # Assign to dataframe
            labels[f'event_occurs_{h_hours}h'] = occurs
            labels[f'event_type_{h_hours}h'] = types
            labels[f'event_magnitude_{h_hours}h'] = mags
            labels[f'event_duration_{h_hours}h'] = durs
            labels[f'event_timing_{h_hours}h'] = timings
            
            event_rate = occurs.mean()
            print(f"✓ {event_rate*100:.2f}% positive samples")
        
        return labels
    
    def _create_per_band_labels(self, labels: pd.DataFrame, per_band_events: dict,
                            horizons_timesteps: list, horizons_hours: list, n_samples: int) -> pd.DataFrame:
        """✨ FIXED: Create per-band magnitude and duration labels (NORMALIZED)"""
        
        print(f"\nCreating PER-BAND labels (NORMALIZED) from {len(per_band_events)} frequency bands...")
        
        for band in self.config.FREQUENCY_BANDS:
            if band not in per_band_events or per_band_events[band].empty:
                print(f"  {band}: No events, using zeros")
                continue
            
            events_df = per_band_events[band]
            print(f"  Processing {band}: {len(events_df)} events...", end=" ")
            
            # ✨ Get normalization factor for this band
            if band in self.band_stats:
                band_std = self.band_stats[band]['std']
            else:
                band_std = 1.0
                print(f"⚠️ No stats for {band}, using std=1.0")
            
            # Prepare band event data
            events_sorted = events_df.sort_values('t1').reset_index(drop=True)
            event_starts = events_sorted['t1'].values.astype(int)
            
            # ✅ Extract magnitudes (RAW)
            magnitudes_raw = np.zeros(len(events_sorted))
            for idx, event in events_sorted.iterrows():
                if event['event_type'] == 'significant' and pd.notna(event.get('∆w_m')):
                    magnitudes_raw[idx] = abs(event['∆w_m'])
                elif event['event_type'] == 'stationary' and pd.notna(event.get('σ_s')):
                    magnitudes_raw[idx] = event['σ_s']
            
            # ✅ IMPROVED: ADAPTIVE NORMALIZATION BASED ON MAGNITUDE SCALE
            raw_median = np.median(magnitudes_raw[magnitudes_raw > 0]) if np.sum(magnitudes_raw > 0) > 0 else 1.0
            
            print(f"median={raw_median:.2f}, ", end="")
            
            if raw_median < 10:
                # Already normalized (0-10 range), use as-is
                print(f"(already normalized) ", end="")
                magnitudes_normalized = magnitudes_raw
            elif raw_median > 100:
                # Very large values (DWT coefficients), normalize by 95th percentile
                print(f"(95th percentile norm) ", end="")
                p95 = np.percentile(magnitudes_raw[magnitudes_raw > 0], 95)
                magnitudes_normalized = magnitudes_raw / (p95 + 1e-8)
                magnitudes_normalized = np.clip(magnitudes_normalized, 0, 3)  # Clip outliers to [0, 3]
            else:
                # Moderate values (10-100 range), normalize by band std
                print(f"(band std={band_std:.2f}) ", end="")
                magnitudes_normalized = magnitudes_raw / (band_std + 1e-8)
            
            # Extract durations
            durations = np.ones(len(events_sorted))
            for idx, event in events_sorted.iterrows():
                if event['event_type'] == 'significant' and pd.notna(event.get('∆t_m')):
                    durations[idx] = event['∆t_m']
                elif event['event_type'] == 'stationary' and pd.notna(event.get('∆t_s')):
                    durations[idx] = event['∆t_s']
            
            # For each horizon, create band-specific labels
            for h_steps, h_hours in zip(horizons_timesteps, horizons_hours):
                mags = np.zeros(n_samples, dtype=np.float32)
                durs = np.zeros(n_samples, dtype=np.float32)
                
                valid_indices = np.arange(0, n_samples - h_steps)
                
                if len(valid_indices) > 0:
                    for idx in valid_indices:
                        future_end = idx + h_steps
                        
                        # Find events in window
                        start_idx = np.searchsorted(event_starts, idx, side='left')
                        end_idx = np.searchsorted(event_starts, future_end, side='right')
                        
                        if start_idx < end_idx:
                            window_mask = (event_starts >= idx) & (event_starts < future_end)
                            window_indices = np.where(window_mask)[0]
                            
                            if len(window_indices) > 0:
                                # Max NORMALIZED magnitude in window for this band
                                window_mags = magnitudes_normalized[window_indices]
                                mags[idx] = np.max(window_mags)
                                
                                # Mean duration in window for this band
                                window_durs = durations[window_indices]
                                durs[idx] = np.mean(window_durs)
                
                # Assign to dataframe
                labels[f'event_magnitude_{h_hours}h_{band}'] = mags
                labels[f'event_duration_{h_hours}h_{band}'] = durs
            
            print(f"✓")
        
        return labels

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: TRAINING WITH STAGED APPROACH (FOR HAWKES)
# ═══════════════════════════════════════════════════════════════════════════

def train_model_staged(model, train_loader, val_loader, criterion, config, device):
    """
    Staged Training for Hawkes + LSTM Hybrid
    
    Stage 1 (5 epochs): Train only Hawkes layer (LSTM frozen)
    Stage 2 (45 epochs): Train everything (end-to-end fine-tuning)
    
    This prevents overfitting by letting Hawkes learn event causality first,
    then fine-tuning the entire model.
    """
    
    print(f"\n{'='*80}")
    print(f" STAGED TRAINING - {config.get_ablation_name()}")
    print(f"{'='*80}\n")
    
    if not config.USE_STAGED_TRAINING or not config.USE_HAWKES_PROCESS:
        print(" Staged training disabled, using standard training...")
        # Fall back to regular training
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=0.02
        )
        return train_model(model, train_loader, val_loader, optimizer, criterion, config, device)
    
    # ========== STAGE 1: TRAIN HAWKES LAYER ONLY ==========
    print(f"{'─'*80}")
    print("STAGE 1: Training Hawkes Layer (LSTM Frozen)")
    print(f"{'─'*80}\n")
    print(f"Epochs: {config.HAWKES_PRETRAIN_EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Weight decay: 0.02\n")
    
    # Freeze LSTM and prediction heads
    for name, param in model.named_parameters():
        if 'hawkes' not in name:
            param.requires_grad = False
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (Stage 1): {trainable:,}\n")
    
    # Optimizer for Hawkes layer only
    optimizer_hawkes = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=0.03
    )
    
    # Train Stage 1
    stage1_losses = []
    for epoch in range(config.HAWKES_PRETRAIN_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = {h: {k: (v.to(device) if not isinstance(v, dict) else 
                              {band: tensor.to(device) for band, tensor in v.items()})
                          for k, v in tasks.items()} 
                      for h, tasks in batch_y.items()}
            
            optimizer_hawkes.zero_grad()
            predictions = model(batch_X)
            loss, _ = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_hawkes.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        stage1_losses.append(epoch_loss)
        print(f"Stage 1 - Epoch {epoch+1}/{config.HAWKES_PRETRAIN_EPOCHS} | Loss: {epoch_loss:.4f}")
    
    print(f"\n✓ Stage 1 complete! Hawkes layer initialized.\n")
    
    # ========== STAGE 2: FINE-TUNE EVERYTHING ==========
    print(f"{'─'*80}")
    print("STAGE 2: End-to-End Fine-Tuning (All Parameters)")
    print(f"{'─'*80}\n")
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (Stage 2): {trainable:,}")
    print(f"Learning rate: {config.LEARNING_RATE * 0.5} (reduced for fine-tuning)")
    print(f"Weight decay: 0.02\n")
    
    # Optimizer for all parameters (lower LR for stability)
    optimizer_all = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE * 0.5,  # Lower LR for fine-tuning
        weight_decay=0.03
    )
    
    # Train Stage 2 with early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    train_losses, val_losses = [], []
    warmup_epochs = 10
    
    remaining_epochs = config.NUM_EPOCHS - config.HAWKES_PRETRAIN_EPOCHS
    
    import time
    
    for epoch in range(remaining_epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = {h: {k: (v.to(device) if not isinstance(v, dict) else 
                              {band: tensor.to(device) for band, tensor in v.items()})
                          for k, v in tasks.items()} 
                      for h, tasks in batch_y.items()}
            
            optimizer_all.zero_grad()
            predictions = model(batch_X)
            loss, _ = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_all.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = {h: {k: (v.to(device) if not isinstance(v, dict) else 
                                  {band: tensor.to(device) for band, tensor in v.items()})
                              for k, v in tasks.items()} 
                          for h, tasks in batch_y.items()}
                predictions = model(batch_X)
                loss, _ = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Early stopping logic
        improvement = ""
        
        if epoch < warmup_epochs:
            if val_loss < best_val_loss:
                improvement = " !!!NEW BEST (WARMUP)!!!"
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            else:
                improvement = " (WARMUP)"
            
            print(f"Stage 2 - Epoch {epoch+1:3d}/{remaining_epochs} | "
                  f"Train: {train_loss:7.4f} | Val: {val_loss:7.4f} | "
                  f"Time: {epoch_time:5.2f}s{improvement}")
        else:
            if val_loss < best_val_loss:
                improvement = " !!!NEW BEST!!!"
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Stage 2 - Epoch {epoch+1:3d}/{remaining_epochs} | "
                  f"Train: {train_loss:7.4f} | Val: {val_loss:7.4f} | "
                  f"Time: {epoch_time:5.2f}s | Patience: {patience_counter}/{config.PATIENCE}{improvement}")
            
            if patience_counter >= config.PATIENCE:
                print(f"\n{'─'*80}")
                print(f"Early stopping at epoch {epoch+1}")
                print(f"{'─'*80}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Training summary
    print(f"\n{'='*80}")
    print(f"✓ Staged Training Complete! Best Val Loss: {best_val_loss:.4f}")
    
    if len(train_losses) > warmup_epochs:
        initial_train_loss = np.mean(train_losses[:3])
        final_train_loss = train_losses[-1]
        initial_val_loss = np.mean(val_losses[:3])
        
        train_improvement = (1 - final_train_loss / initial_train_loss) * 100
        val_improvement = (1 - best_val_loss / initial_val_loss) * 100
        final_gap = val_losses[-1] - train_losses[-1]
        overfitting = "Yes" if final_gap > train_losses[-1] * 0.5 else "No"
        
        print(f"Training improvement: {train_improvement:.1f}%")
        print(f"Validation improvement: {val_improvement:.1f}%")
        print(f"Final train-val gap: {final_gap:.4f}")
        print(f"Overfitting detected: {overfitting}")
    
    print(f"{'='*80}\n")
    
    # Combine stage 1 and stage 2 losses for visualization
    all_train_losses = stage1_losses + train_losses
    all_val_losses = [stage1_losses[-1]] * len(stage1_losses) + val_losses  # Placeholder for stage 1 val
    
    return model, all_train_losses, all_val_losses


# Keep the regular train_model with freq-Aware LSTM function for when staged training is disabled
def train_model(model, train_loader, val_loader, optimizer, criterion, config, device):
    """ Train model with live epoch tracking + EARLY STOPPING WARMUP + GRADIENT CLIPPING"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING - {config.get_ablation_name()}")
    print(f"{'='*80}\n")
    print(f"Total epochs: {config.NUM_EPOCHS}, Patience: {config.PATIENCE}")
    print(f" Early stopping warmup: 10 epochs")  
    print(f"{'─'*80}\n")
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    train_losses, val_losses = [], []
    
    # PHASE 1: Early stopping warmup
    warmup_epochs = 10
    
    import time
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()
        
        # ========== TRAINING PHASE ==========
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = {h: {k: (v.to(device) if not isinstance(v, dict) else 
                              {band: tensor.to(device) for band, tensor in v.items()})
                          for k, v in tasks.items()} 
                      for h, tasks in batch_y.items()}
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss, _ = criterion(predictions, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # ========== VALIDATION PHASE ==========
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = {h: {k: (v.to(device) if not isinstance(v, dict) else 
                                  {band: tensor.to(device) for band, tensor in v.items()})
                              for k, v in tasks.items()} 
                          for h, tasks in batch_y.items()}
                predictions = model(batch_X)
                loss, _ = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # ========== EARLY STOPPING WITH WARMUP ==========
        improvement = ""
        
        if epoch < warmup_epochs:
            # During warmup: just track best, don't check patience
            if val_loss < best_val_loss:
                improvement = " !!!NEW BEST (WARMUP)!!!"
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            else:
                improvement = " (WARMUP - NO PATIENCE CHECK)"
            
            print(f"Epoch {epoch+1:3d}/{config.NUM_EPOCHS} | "
                  f"Train: {train_loss:7.4f} | Val: {val_loss:7.4f} | "
                  f"Time: {epoch_time:5.2f}s{improvement}")
        
        else:
            # After warmup: normal early stopping
            if val_loss < best_val_loss:
                improvement = " !!!NEW BEST!!!"
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch+1:3d}/{config.NUM_EPOCHS} | "
                  f"Train: {train_loss:7.4f} | Val: {val_loss:7.4f} | "
                  f"Time: {epoch_time:5.2f}s | Patience: {patience_counter}/{config.PATIENCE}{improvement}")
            
            # Check early stopping
            if patience_counter >= config.PATIENCE:
                print(f"\n{'─'*80}")
                print(f"Early stopping at epoch {epoch+1}")
                print(f"{'─'*80}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Training summary with diagnostics
    print(f"\n{'='*80}")
    print(f"✓ Training complete! Best Val Loss: {best_val_loss:.4f}")
    
    # Calculate training quality metrics
    if len(train_losses) > warmup_epochs:
        initial_train_loss = np.mean(train_losses[:3])
        final_train_loss = train_losses[-1]
        initial_val_loss = np.mean(val_losses[:3])
        final_val_loss = val_losses[-1]
        
        train_improvement = (1 - final_train_loss / initial_train_loss) * 100
        val_improvement = (1 - best_val_loss / initial_val_loss) * 100
        
        # Overfitting check
        final_gap = final_val_loss - final_train_loss
        overfitting = "Yes" if final_gap > final_train_loss * 0.5 else "No"
        
        print(f"Training improvement: {train_improvement:.1f}%")
        print(f"Validation improvement: {val_improvement:.1f}%")
        print(f"Final train-val gap: {final_gap:.4f}")
        print(f"Overfitting detected: {overfitting}")
    
    print(f"{'='*80}\n")
    return model, train_losses, val_losses

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10B: ENHANCED EVALUATION ( WITH RMSE, NRMSE, MAPE, R²)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, test_loader, config, device):
    """
    Enhanced Evaluation with Comprehensive Metrics
    
    Per-Band Metrics:
    - R² (coefficient of determination)
    - RMSE (root mean squared error)
    - NRMSE (normalized RMSE, %)
    - MAE (mean absolute error)
    - MAPE (mean absolute percentage error, %)
    - Correlation coefficient
    
    Aggregate Metrics:
    - Weighted combinations across bands
    - Event detection (F1, precision, recall, AUC)
    - Type classification accuracy
    """
    
    print(f"\n{'='*80}")
    print("EVALUATION ( WITH COMPREHENSIVE METRICS)")
    print(f"{'='*80}\n")
    
    model.eval()
    horizons = config.PREDICTION_HORIZONS_HOURS
    use_freq_aware = config.USE_FREQUENCY_AWARE_PREDICTION
    
    # Initialize collectors
    all_preds = {h: {'occurs': [], 'type': [], 'timing': []} for h in horizons}
    all_targets = {h: {'occurs': [], 'type': [], 'timing': []} for h in horizons}
    
    if use_freq_aware:
        for h in horizons:
            all_preds[h]['magnitude_per_band'] = {band: [] for band in config.FREQUENCY_BANDS}
            all_preds[h]['duration_per_band'] = {band: [] for band in config.FREQUENCY_BANDS}
            all_targets[h]['magnitude_per_band'] = {band: [] for band in config.FREQUENCY_BANDS}
            all_targets[h]['duration_per_band'] = {band: [] for band in config.FREQUENCY_BANDS}
    else:
        for h in horizons:
            all_preds[h]['magnitude'] = []
            all_preds[h]['duration'] = []
            all_targets[h]['magnitude'] = []
            all_targets[h]['duration'] = []
    
    # Collect predictions
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            predictions = model(batch_X)
            
            for h in horizons:
                # Global tasks
                all_preds[h]['occurs'].append(predictions[h]['occurs'].cpu().numpy())
                all_preds[h]['type'].append(predictions[h]['type'].cpu().numpy())
                all_preds[h]['timing'].append(predictions[h]['timing'].cpu().numpy())
                
                all_targets[h]['occurs'].append(batch_y[h]['occurs'].cpu().numpy())
                all_targets[h]['type'].append(batch_y[h]['type'].cpu().numpy())
                all_targets[h]['timing'].append(batch_y[h]['timing'].cpu().numpy())
                
                # Per-band or single
                if use_freq_aware:
                    for band in config.FREQUENCY_BANDS:
                        all_preds[h]['magnitude_per_band'][band].append(
                            predictions[h]['magnitude_per_band'][band].cpu().numpy())
                        all_preds[h]['duration_per_band'][band].append(
                            predictions[h]['duration_per_band'][band].cpu().numpy())
                        all_targets[h]['magnitude_per_band'][band].append(
                            batch_y[h]['magnitude_per_band'][band].cpu().numpy())
                        all_targets[h]['duration_per_band'][band].append(
                            batch_y[h]['duration_per_band'][band].cpu().numpy())
                else:
                    all_preds[h]['magnitude'].append(predictions[h]['magnitude'].cpu().numpy())
                    all_preds[h]['duration'].append(predictions[h]['duration'].cpu().numpy())
                    all_targets[h]['magnitude'].append(batch_y[h]['magnitude'].cpu().numpy())
                    all_targets[h]['duration'].append(batch_y[h]['duration'].cpu().numpy())
    
    # Concatenate all batches
    for h in horizons:
        all_preds[h]['occurs'] = np.concatenate(all_preds[h]['occurs'])
        all_preds[h]['type'] = np.concatenate(all_preds[h]['type'])
        all_preds[h]['timing'] = np.concatenate(all_preds[h]['timing'])
        all_targets[h]['occurs'] = np.concatenate(all_targets[h]['occurs'])
        all_targets[h]['type'] = np.concatenate(all_targets[h]['type'])
        all_targets[h]['timing'] = np.concatenate(all_targets[h]['timing'])
        
        if use_freq_aware:
            for band in config.FREQUENCY_BANDS:
                all_preds[h]['magnitude_per_band'][band] = np.concatenate(all_preds[h]['magnitude_per_band'][band])
                all_preds[h]['duration_per_band'][band] = np.concatenate(all_preds[h]['duration_per_band'][band])
                all_targets[h]['magnitude_per_band'][band] = np.concatenate(all_targets[h]['magnitude_per_band'][band])
                all_targets[h]['duration_per_band'][band] = np.concatenate(all_targets[h]['duration_per_band'][band])
        else:
            all_preds[h]['magnitude'] = np.concatenate(all_preds[h]['magnitude'])
            all_preds[h]['duration'] = np.concatenate(all_preds[h]['duration'])
            all_targets[h]['magnitude'] = np.concatenate(all_targets[h]['magnitude'])
            all_targets[h]['duration'] = np.concatenate(all_targets[h]['duration'])
    
    #  Helper function for comprehensive metrics
    def compute_regression_metrics(y_true, y_pred, metric_name=""):
        """Compute comprehensive regression metrics"""
        metrics = {}
        
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                'r2': 0.0, 'rmse': 0.0, 'nrmse': 0.0, 'mae': 0.0, 
                'mape': 0.0, 'correlation': 0.0
            }
        
        # R² score
        try:
            metrics['r2'] = r2_score(y_true, y_pred)
        except:
            metrics['r2'] = -999.0
        
        # RMSE
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # NRMSE (normalized by range, expressed as percentage)
        y_range = y_true.max() - y_true.min()
        if y_range > 1e-8:
            metrics['nrmse'] = (metrics['rmse'] / y_range) * 100
        else:
            metrics['nrmse'] = 0.0
        
        # MAE
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # MAPE (mean absolute percentage error)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
        metrics['mape'] = np.clip(mape, 0, 1000)  # Cap at 1000% for outliers
        
        # Pearson correlation coefficient
        try:
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            metrics['correlation'] = corr if not np.isnan(corr) else 0.0
        except:
            metrics['correlation'] = 0.0
        
        return metrics
    
    # Calculate metrics for each horizon
    results = {}
    
    for h in horizons:
        h_timesteps = config.hours_to_timesteps(h)
        print(f"\n{h}h Horizon ({h_timesteps} timesteps):")
        print(f"{'─'*80}")
        
        # ========== OCCURRENCE METRICS ==========
        pred_occurs = (all_preds[h]['occurs'].flatten() > 0.5).astype(int)
        true_occurs = all_targets[h]['occurs'].astype(int)
        
        f1 = f1_score(true_occurs, pred_occurs, zero_division=0)
        precision = precision_score(true_occurs, pred_occurs, zero_division=0)
        recall = recall_score(true_occurs, pred_occurs, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(true_occurs, all_preds[h]['occurs'].flatten())
        except:
            roc_auc = 0.0
        
        print(f"\n  Event Detection:")
        print(f"    F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, AUC={roc_auc:.4f}")
        
        # Type accuracy
        pred_type = np.argmax(all_preds[h]['type'], axis=1)
        true_type = all_targets[h]['type'].astype(int)
        type_acc = (pred_type == true_type).mean()
        print(f"    Type Accuracy={type_acc:.4f}")
        
        # Event mask (only evaluate magnitude/duration for actual events)
        event_mask = true_occurs == 1
        n_events = event_mask.sum()
        
        print(f"\n   Regression Metrics (on {n_events} events):")
        
        # ========== PER-BAND COMPREHENSIVE METRICS ==========
        if use_freq_aware:
            print(f"\n  Per-Band Magnitude Metrics:")
            print(f"  {'Band':<15} {'R²':>8} {'RMSE':>8} {'NRMSE%':>8} {'MAE':>8} {'MAPE%':>8} {'Corr':>8}")
            print(f"  {'-'*75}")
            
            mag_metrics_per_band = {}
            dur_metrics_per_band = {}
            
            for band in config.FREQUENCY_BANDS:
                if n_events > 0:
                    # Magnitude metrics
                    pred_mag_band = all_preds[h]['magnitude_per_band'][band].flatten()[event_mask]
                    true_mag_band = all_targets[h]['magnitude_per_band'][band][event_mask]
                    
                    mag_metrics = compute_regression_metrics(true_mag_band, pred_mag_band, f"mag_{band}")
                    mag_metrics_per_band[band] = mag_metrics
                    
                    # Duration metrics
                    pred_dur_band = all_preds[h]['duration_per_band'][band].flatten()[event_mask]
                    true_dur_band = all_targets[h]['duration_per_band'][band][event_mask]
                    
                    dur_metrics = compute_regression_metrics(true_dur_band, pred_dur_band, f"dur_{band}")
                    dur_metrics_per_band[band] = dur_metrics
                    
                    # Print magnitude metrics
                    print(f"  {band:<15} "
                          f"{mag_metrics['r2']:>8.4f} "
                          f"{mag_metrics['rmse']:>8.4f} "
                          f"{mag_metrics['nrmse']:>8.2f} "
                          f"{mag_metrics['mae']:>8.4f} "
                          f"{mag_metrics['mape']:>8.2f} "
                          f"{mag_metrics['correlation']:>8.4f}")
                else:
                    mag_metrics_per_band[band] = {
                        'r2': 0.0, 'rmse': 0.0, 'nrmse': 0.0, 
                        'mae': 0.0, 'mape': 0.0, 'correlation': 0.0
                    }
                    dur_metrics_per_band[band] = {
                        'r2': 0.0, 'rmse': 0.0, 'nrmse': 0.0, 
                        'mae': 0.0, 'mape': 0.0, 'correlation': 0.0
                    }
            
            #  Per-Band Duration Metrics
            print(f"\n  Per-Band Duration Metrics:")
            print(f"  {'Band':<15} {'R²':>8} {'RMSE':>8} {'NRMSE%':>8} {'MAE':>8} {'MAPE%':>8} {'Corr':>8}")
            print(f"  {'-'*75}")
            
            for band in config.FREQUENCY_BANDS:
                if n_events > 0:
                    dur_m = dur_metrics_per_band[band]
                    print(f"  {band:<15} "
                          f"{dur_m['r2']:>8.4f} "
                          f"{dur_m['rmse']:>8.4f} "
                          f"{dur_m['nrmse']:>8.2f} "
                          f"{dur_m['mae']:>8.4f} "
                          f"{dur_m['mape']:>8.2f} "
                          f"{dur_m['correlation']:>8.4f}")
            
            # ========== AGGREGATE METRICS (WEIGHTED) ==========
            band_weights = {
                'approximation': 0.5,
                'details_4': 0.8,
                'details_3': 1.0,  # Main ramp band
                'details_2': 0.8,
                'details_1': 0.5
            }
            
            if n_events > 0:
                # Weighted aggregation for magnitude
                pred_mag_agg = np.zeros_like(all_preds[h]['magnitude_per_band'][config.FREQUENCY_BANDS[0]].flatten())
                true_mag_agg = np.zeros_like(all_targets[h]['magnitude_per_band'][config.FREQUENCY_BANDS[0]])
                
                total_weight = sum(band_weights.values())
                
                for band in config.FREQUENCY_BANDS:
                    weight = band_weights.get(band, 1.0) / total_weight
                    pred_mag_agg += weight * all_preds[h]['magnitude_per_band'][band].flatten()
                    true_mag_agg += weight * all_targets[h]['magnitude_per_band'][band]
                
                # Compute aggregate magnitude metrics
                agg_mag_metrics = compute_regression_metrics(
                    true_mag_agg[event_mask], 
                    pred_mag_agg[event_mask],
                    "aggregate_magnitude"
                )
                
                # Weighted aggregation for duration
                pred_dur_agg = np.zeros_like(all_preds[h]['duration_per_band'][config.FREQUENCY_BANDS[0]].flatten())
                true_dur_agg = np.zeros_like(all_targets[h]['duration_per_band'][config.FREQUENCY_BANDS[0]])
                
                for band in config.FREQUENCY_BANDS:
                    weight = band_weights.get(band, 1.0) / total_weight
                    pred_dur_agg += weight * all_preds[h]['duration_per_band'][band].flatten()
                    true_dur_agg += weight * all_targets[h]['duration_per_band'][band]
                
                # Compute aggregate duration metrics
                agg_dur_metrics = compute_regression_metrics(
                    true_dur_agg[event_mask], 
                    pred_dur_agg[event_mask],
                    "aggregate_duration"
                )
                
                # Timing metrics
                timing_metrics = compute_regression_metrics(
                    all_targets[h]['timing'][event_mask],
                    all_preds[h]['timing'].flatten()[event_mask],
                    "timing"
                )
            else:
                agg_mag_metrics = {'r2': 0.0, 'rmse': 0.0, 'nrmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'correlation': 0.0}
                agg_dur_metrics = {'r2': 0.0, 'rmse': 0.0, 'nrmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'correlation': 0.0}
                timing_metrics = {'r2': 0.0, 'rmse': 0.0, 'nrmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'correlation': 0.0}
                pred_mag_agg = all_preds[h]['magnitude_per_band'][config.FREQUENCY_BANDS[0]].flatten()
                true_mag_agg = all_targets[h]['magnitude_per_band'][config.FREQUENCY_BANDS[0]]
            
            # Print aggregate metrics
            print(f"\n    Aggregate Metrics (weighted across bands):")
            print(f"\n    Magnitude:")
            print(f"      R²={agg_mag_metrics['r2']:>8.4f}, "
                  f"RMSE={agg_mag_metrics['rmse']:>8.4f}, "
                  f"NRMSE={agg_mag_metrics['nrmse']:>7.2f}%, "
                  f"MAE={agg_mag_metrics['mae']:>8.4f}")
            print(f"      MAPE={agg_mag_metrics['mape']:>7.2f}%, "
                  f"Correlation={agg_mag_metrics['correlation']:>8.4f}")
            
            print(f"\n    Duration:")
            print(f"      R²={agg_dur_metrics['r2']:>8.4f}, "
                  f"RMSE={agg_dur_metrics['rmse']:>8.4f}, "
                  f"NRMSE={agg_dur_metrics['nrmse']:>7.2f}%, "
                  f"MAE={agg_dur_metrics['mae']:>8.4f}")
            print(f"      MAPE={agg_dur_metrics['mape']:>7.2f}%, "
                  f"Correlation={agg_dur_metrics['correlation']:>8.4f}")
            
            print(f"\n    Timing:")
            print(f"      MAE={timing_metrics['mae']:>8.4f}, "
                  f"RMSE={timing_metrics['rmse']:>8.4f}, "
                  f"Correlation={timing_metrics['correlation']:>8.4f}")
            
        else:
            # ========== SINGLE MAGNITUDE/DURATION METRICS ==========
            if n_events > 0:
                # Magnitude
                pred_mag = all_preds[h]['magnitude'].flatten()[event_mask]
                true_mag = all_targets[h]['magnitude'][event_mask]
                agg_mag_metrics = compute_regression_metrics(true_mag, pred_mag, "magnitude")
                
                # Duration
                pred_dur = all_preds[h]['duration'].flatten()[event_mask]
                true_dur = all_targets[h]['duration'][event_mask]
                agg_dur_metrics = compute_regression_metrics(true_dur, pred_dur, "duration")
                
                # Timing
                timing_metrics = compute_regression_metrics(
                    all_targets[h]['timing'][event_mask],
                    all_preds[h]['timing'].flatten()[event_mask],
                    "timing"
                )
                
                pred_mag_agg = all_preds[h]['magnitude'].flatten()
                true_mag_agg = all_targets[h]['magnitude']
            else:
                agg_mag_metrics = {'r2': 0.0, 'rmse': 0.0, 'nrmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'correlation': 0.0}
                agg_dur_metrics = {'r2': 0.0, 'rmse': 0.0, 'nrmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'correlation': 0.0}
                timing_metrics = {'r2': 0.0, 'rmse': 0.0, 'nrmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'correlation': 0.0}
                pred_mag_agg = all_preds[h]['magnitude'].flatten()
                true_mag_agg = all_targets[h]['magnitude']
            
            mag_metrics_per_band = {}
            dur_metrics_per_band = {}
            
            print(f"\n  Single Metrics:")
            print(f"    Magnitude: R²={agg_mag_metrics['r2']:.4f}, RMSE={agg_mag_metrics['rmse']:.4f}, "
                  f"NRMSE={agg_mag_metrics['nrmse']:.2f}%, MAE={agg_mag_metrics['mae']:.4f}")
            print(f"    Duration: R²={agg_dur_metrics['r2']:.4f}, RMSE={agg_dur_metrics['rmse']:.4f}, "
                  f"MAE={agg_dur_metrics['mae']:.4f}")
            print(f"    Timing: MAE={timing_metrics['mae']:.4f}")
        
        # Store results
        results[h] = {
            # Event detection
            'f1': f1, 
            'precision': precision, 
            'recall': recall, 
            'roc_auc': roc_auc, 
            'type_accuracy': type_acc,
            
            # Per-band comprehensive metrics
            'magnitude_metrics_per_band': mag_metrics_per_band,
            'duration_metrics_per_band': dur_metrics_per_band,
            
            # Aggregate metrics
            'magnitude_r2': agg_mag_metrics['r2'],
            'magnitude_rmse': agg_mag_metrics['rmse'],
            'magnitude_nrmse': agg_mag_metrics['nrmse'],
            'magnitude_mae': agg_mag_metrics['mae'],
            'magnitude_mape': agg_mag_metrics['mape'],
            'magnitude_correlation': agg_mag_metrics['correlation'],
            
            'duration_r2': agg_dur_metrics['r2'],
            'duration_rmse': agg_dur_metrics['rmse'],
            'duration_nrmse': agg_dur_metrics['nrmse'],
            'duration_mae': agg_dur_metrics['mae'],
            'duration_mape': agg_dur_metrics['mape'],
            'duration_correlation': agg_dur_metrics['correlation'],
            
            'timing_mae': timing_metrics['mae'],
            'timing_rmse': timing_metrics['rmse'],
            'timing_correlation': timing_metrics['correlation'],
            
            # Raw predictions and targets
            'predictions': all_preds[h], 
            'targets': all_targets[h]
        }
    
    print(f"\n{'='*80}\n")
    return results

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11: ENHANCED VISUALIZER (🆕 WITH DEBUGGING & ANALYSIS PLOTS)
# ═══════════════════════════════════════════════════════════════════════════

class EnhancedDWTVisualizer:
    """Comprehensive visualization suite with debugging tools"""

    def __init__(self, config):
        self.config = config

    def plot_all_diagnostics(self, results: dict, reconstruction_results: dict,
                         train_losses: list, val_losses: list,
                         dwt_decomposition: dict, original_signal: np.ndarray,
                         save_dir: str,
                         events_df: pd.DataFrame = None):
        """Generate ALL diagnostic plots in one call"""

        import os
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print("GENERATING DIAGNOSTIC VISUALIZATIONS")
        print(f"{'='*80}\n")

        # 1. DWT Decomposition
        print(" DWT Frequency Decomposition...")
        self.plot_dwt_decomposition(dwt_decomposition, original_signal,
                                   f"{save_dir}/dwt_decomposition.png")

        # 2. Training History
        print(" Training History...")
        self.plot_training_history(train_losses, val_losses,
                                   f"{save_dir}/training_history.png")

        # 3. Per-Band Prediction Analysis
        print("  Per-Band Prediction Analysis...")
        self.plot_perband_analysis(results,
                                   f"{save_dir}/perband_analysis.png")

        # 4. Prediction vs Target Scatter (all bands)
        print("  Prediction Quality Scatter...")
        self.plot_prediction_scatter(results,
                                     f"{save_dir}/prediction_scatter.png")

        # 5. Reconstruction Comparison
        print(" Reconstruction Comparison...")
        self.plot_reconstruction_comparison(reconstruction_results,
                                           f"{save_dir}/reconstruction.png")

        # 6. Comprehensive Performance Summary
        print(" Performance Summary...")
        self.plot_performance_summary(results, reconstruction_results,
                                 f"{save_dir}/performance_summary.png")
    
        # 7. Events on Time-Series (Type/Direction/Duration View)
        print(" Events on Time-Series (Comprehensive)...")
        if events_df is not None and not events_df.empty:
            self.plot_events_on_timeseries(results, original_signal, events_df,
                                           f"{save_dir}/events_on_timeseries.png")
        else:
            print("       ⚠️  Skipped (no events_df)")
    
        # 8. Zoomed Event Comparison
        print("  Zoomed Event Comparison...")
        if events_df is not None and not events_df.empty:
            self.plot_zoomed_event_comparison(results, original_signal, events_df,
                                              f"{save_dir}/zoomed_events.png")
        else:
            print("       ⚠️  Skipped (no events_df)")

        
        print("  More analysis on predicted_vs_actual events...")
        self.plot_predicted_vs_actual_events(results,
                                        f"{save_dir}/predicted_vs_actual_events.png")

        print("\n  📊 Generating detailed hybrid reconstruction visualizations...")

        # Get events dataframe for overlay plots
        events_df = None  # You'll need to pass this if available

        # Create reconstructor instance to access plotting methods
        reconstructor = iDWTReconstructor(self.config)

        # Call the detailed plotting methods
        reconstructor.plot_hybrid_reconstruction_detailed(
            reconstruction_results,
            save_dir
        )

        # Only call events overlay if events_df is available
        if events_df is not None:
            reconstructor.plot_reconstruction_with_events_overlay(
                reconstruction_results,
                results,
                original_signal,
                events_df,
                save_dir
            )

        print(f"\n✓ All diagnostics saved to: {save_dir}/")
        print(f"{'='*80}\n")

    def plot_dwt_decomposition(self, dwt_decomposition: dict, original_signal: np.ndarray,
                              save_path: str):
        """ Visualize DWT frequency decomposition"""

        decomposer = DWTDecomposer(self.config)

        # Reconstruct individual bands
        bands = {
            'Original': original_signal[:2000],
            'Approximation (Trend)': decomposer.reconstruct_from_bands(dwt_decomposition, ['approximation'])[:2000],
            'Details 4 (Slow)': decomposer.reconstruct_from_bands(dwt_decomposition, ['details_4'])[:2000],
            'Details 3 (Ramps)': decomposer.reconstruct_from_bands(dwt_decomposition, ['details_3'])[:2000],
            'Details 2 (Fast)': decomposer.reconstruct_from_bands(dwt_decomposition, ['details_2'])[:2000],
            'Details 1 (Noise)': decomposer.reconstruct_from_bands(dwt_decomposition, ['details_1'])[:2000]
        }

        fig, axes = plt.subplots(6, 1, figsize=(20, 14))
        x = np.arange(2000)

        colors = ['black', 'blue', 'green', 'red', 'orange', 'purple']

        for idx, (band_name, signal) in enumerate(bands.items()):
            axes[idx].plot(x, signal, color=colors[idx], linewidth=1.5, alpha=0.8)
            axes[idx].set_ylabel(band_name, fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

            # Add statistics
            stats_text = f"Mean: {signal.mean():.2f} | Std: {signal.std():.2f} | Range: [{signal.min():.2f}, {signal.max():.2f}]"
            axes[idx].text(0.02, 0.95, stats_text, transform=axes[idx].transAxes,
                          fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        axes[-1].set_xlabel('Time (samples)', fontsize=12)
        axes[0].set_title('DWT Multi-Resolution Decomposition (Frequency Bands)',
                         fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_training_history(self, train_losses: list, val_losses: list, save_path: str):
        """ Training and validation loss curves with diagnostics"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        epochs = np.arange(1, len(train_losses) + 1)

        # Find best epoch
        best_epoch = np.argmin(val_losses) + 1
        best_val_loss = val_losses[best_epoch - 1]

        # SUBPLOT 1: Loss curves (linear)
        axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.8)
        axes[0, 0].plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss', alpha=0.8)
        axes[0, 0].axvline(x=best_epoch, color='g', linestyle='--', linewidth=2,
                          label=f'Best Epoch ({best_epoch})')
        axes[0, 0].scatter([best_epoch], [best_val_loss], color='green', s=100,
                          zorder=5, marker='*')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training History (Linear Scale)', fontsize=13, fontweight='bold')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)

        # SUBPLOT 2: Loss curves (log scale)
        axes[0, 1].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.8)
        axes[0, 1].plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss', alpha=0.8)
        axes[0, 1].axvline(x=best_epoch, color='g', linestyle='--', linewidth=2,
                          label=f'Best Epoch ({best_epoch})')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Loss (log scale)', fontsize=12)
        axes[0, 1].set_title('Training History (Log Scale)', fontsize=13, fontweight='bold')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3, which='both')

        # SUBPLOT 3: Train-Val Gap (overfitting detector)
        gap = np.array(val_losses) - np.array(train_losses)
        axes[1, 0].plot(epochs, gap, 'purple', linewidth=2, label='Val - Train Gap')
        axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
        axes[1, 0].fill_between(epochs, 0, gap, where=(gap > 0), alpha=0.3, color='red',
                               label='Overfitting Region')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss Gap', fontsize=12)
        axes[1, 0].set_title('Overfitting Detection (Val - Train)', fontsize=13, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)

        # SUBPLOT 4: Loss Reduction Rate
        train_reduction = -np.diff(train_losses)
        val_reduction = -np.diff(val_losses)
        axes[1, 1].plot(epochs[1:], train_reduction, 'b-', linewidth=2,
                       label='Train Reduction', alpha=0.8)
        axes[1, 1].plot(epochs[1:], val_reduction, 'r-', linewidth=2,
                       label='Val Reduction', alpha=0.8)
        axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Loss Reduction Rate', fontsize=12)
        axes[1, 1].set_title('Learning Rate (Loss Reduction per Epoch)', fontsize=13, fontweight='bold')
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)

        # Add summary statistics
        final_train = train_losses[-1]
        final_val = val_losses[-1]
        improvement = (1 - best_val_loss / val_losses[0]) * 100

        summary_text = (
            f"Best Epoch: {best_epoch}/{len(train_losses)}\n"
            f"Best Val Loss: {best_val_loss:.4f}\n"
            f"Final Train Loss: {final_train:.4f}\n"
            f"Final Val Loss: {final_val:.4f}\n"
            f"Improvement: {improvement:.1f}%\n"
            f"Overfitting: {'Yes' if final_val > final_train * 1.5 else 'No'}"
        )

        fig.text(0.98, 0.98, summary_text, transform=fig.transFigure,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_perband_analysis(self, results: dict, save_path: str):
        """ Per-band prediction quality across all horizons"""

        horizons = self.config.PREDICTION_HORIZONS_HOURS
        bands = self.config.FREQUENCY_BANDS

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

       # Collect R² scores
        mag_r2_matrix = np.zeros((len(horizons), len(bands)))
        dur_r2_matrix = np.zeros((len(horizons), len(bands)))

        for i, h in enumerate(horizons):
            if 'magnitude_metrics_per_band' in results[h]:
                for j, band in enumerate(bands):
                    # Extract 'r2' from the metrics dict
                    mag_metrics = results[h]['magnitude_metrics_per_band'].get(band, {})
                    dur_metrics = results[h]['duration_metrics_per_band'].get(band, {})

                    # Get R² value, default to -999 if missing
                    mag_r2_matrix[i, j] = mag_metrics.get('r2', -999) if isinstance(mag_metrics, dict) else -999
                    dur_r2_matrix[i, j] = dur_metrics.get('r2', -999) if isinstance(dur_metrics, dict) else -999

        # Replace -999 with NaN for visualization
        mag_r2_matrix[mag_r2_matrix == -999] = np.nan
        dur_r2_matrix[dur_r2_matrix == -999] = np.nan

        # SUBPLOT 1: Magnitude R² Heatmap
        im1 = axes[0, 0].imshow(mag_r2_matrix, cmap='RdYlGn', aspect='auto',
                               vmin=-0.5, vmax=0.5, interpolation='nearest')
        axes[0, 0].set_xticks(np.arange(len(bands)))
        axes[0, 0].set_yticks(np.arange(len(horizons)))
        axes[0, 0].set_xticklabels(bands, rotation=45, ha='right')
        axes[0, 0].set_yticklabels([f'{h}h' for h in horizons])
        axes[0, 0].set_title('Magnitude R² Score (Per Band, Per Horizon)',
                            fontsize=13, fontweight='bold')

        # Add text annotations
        for i in range(len(horizons)):
            for j in range(len(bands)):
                if not np.isnan(mag_r2_matrix[i, j]):
                    color = 'white' if mag_r2_matrix[i, j] < -0.2 or mag_r2_matrix[i, j] > 0.3 else 'black'
                    axes[0, 0].text(j, i, f'{mag_r2_matrix[i, j]:.2f}',
                                   ha='center', va='center', color=color, fontsize=9, fontweight='bold')

        plt.colorbar(im1, ax=axes[0, 0], label='R² Score')

        # SUBPLOT 2: Duration R² Heatmap
        im2 = axes[0, 1].imshow(dur_r2_matrix, cmap='RdYlGn', aspect='auto',
                               vmin=-0.5, vmax=0.5, interpolation='nearest')
        axes[0, 1].set_xticks(np.arange(len(bands)))
        axes[0, 1].set_yticks(np.arange(len(horizons)))
        axes[0, 1].set_xticklabels(bands, rotation=45, ha='right')
        axes[0, 1].set_yticklabels([f'{h}h' for h in horizons])
        axes[0, 1].set_title('Duration R² Score (Per Band, Per Horizon)',
                            fontsize=13, fontweight='bold')

        for i in range(len(horizons)):
            for j in range(len(bands)):
                if not np.isnan(dur_r2_matrix[i, j]):
                    color = 'white' if dur_r2_matrix[i, j] < -0.2 or dur_r2_matrix[i, j] > 0.3 else 'black'
                    axes[0, 1].text(j, i, f'{dur_r2_matrix[i, j]:.2f}',
                                   ha='center', va='center', color=color, fontsize=9, fontweight='bold')

        plt.colorbar(im2, ax=axes[0, 1], label='R² Score')

        # SUBPLOT 3: Average R² per Band (across all horizons)
        avg_mag_r2 = np.nanmean(mag_r2_matrix, axis=0)
        avg_dur_r2 = np.nanmean(dur_r2_matrix, axis=0)

        x_pos = np.arange(len(bands))
        width = 0.35

        bars1 = axes[1, 0].bar(x_pos - width/2, avg_mag_r2, width,
                              label='Magnitude R²', color='steelblue', alpha=0.8)
        bars2 = axes[1, 0].bar(x_pos + width/2, avg_dur_r2, width,
                              label='Duration R²', color='coral', alpha=0.8)

        axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
        axes[1, 0].set_ylabel('Average R² Score', fontsize=12)
        axes[1, 0].set_xlabel('Frequency Band', fontsize=12)
        axes[1, 0].set_title('Average Prediction Quality per Band', fontsize=13, fontweight='bold')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(bands, rotation=45, ha='right')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                           fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                           fontsize=9)

        # SUBPLOT 4: R² progression across horizons (per band)
        for j, band in enumerate(bands):
            mag_progression = mag_r2_matrix[:, j]
            if not np.all(np.isnan(mag_progression)):
                axes[1, 1].plot(horizons, mag_progression, marker='o', linewidth=2,
                               label=band, alpha=0.8)

        axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
        axes[1, 1].set_xlabel('Prediction Horizon (hours)', fontsize=12)
        axes[1, 1].set_ylabel('Magnitude R²', fontsize=12)
        axes[1, 1].set_title('R² Progression Across Horizons (Magnitude)',
                            fontsize=13, fontweight='bold')
        axes[1, 1].legend(fontsize=10, loc='best')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_prediction_scatter(self, results: dict, save_path: str):
        """ Scatter plots: Predicted vs Actual (all bands, all horizons)"""

        horizons = self.config.PREDICTION_HORIZONS_HOURS
        bands = self.config.FREQUENCY_BANDS

        fig, axes = plt.subplots(len(horizons), len(bands),
                                figsize=(len(bands)*4, len(horizons)*4))

        for i, h in enumerate(horizons):
            for j, band in enumerate(bands):
                ax = axes[i, j] if len(horizons) > 1 else axes[j]

                # Get predictions and targets for events only
                occurs = results[h]['targets']['occurs'].astype(int)
                event_mask = occurs == 1

                if event_mask.sum() > 10:
                    pred_mag = results[h]['predictions']['magnitude_per_band'][band].flatten()[event_mask]
                    true_mag = results[h]['targets']['magnitude_per_band'][band][event_mask]

                    # Scatter plot
                    ax.scatter(true_mag, pred_mag, alpha=0.3, s=20, color='blue')

                    # Perfect prediction line
                    min_val = min(true_mag.min(), pred_mag.min())
                    max_val = max(true_mag.max(), pred_mag.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)

                    # Calculate R²
                    try:
                        r2 = r2_score(true_mag, pred_mag)
                        corr = np.corrcoef(true_mag, pred_mag)[0, 1]
                    except:
                        r2 = -999
                        corr = 0

                    # Title and labels
                    color = 'green' if r2 > 0.2 else ('orange' if r2 > 0 else 'red')
                    ax.set_title(f'{h}h - {band}\nR²={r2:.3f}, r={corr:.3f}',
                                fontsize=10, color=color, fontweight='bold')
                    ax.set_xlabel('True Magnitude', fontsize=9)
                    ax.set_ylabel('Predicted Magnitude', fontsize=9)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'Not enough\nevents\n({event_mask.sum()})',
                           ha='center', va='center', transform=ax.transAxes, fontsize=10)
                    ax.set_title(f'{h}h - {band}', fontsize=10)
                    ax.axis('off')

        plt.suptitle('Predicted vs Actual Magnitude (Per Band, Per Horizon)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_performance_summary(self, results: dict, reconstruction_results: dict,
                            save_path: str):
        """ IMPROVED: Comprehensive performance summary with better spacing"""
    
        horizons = self.config.PREDICTION_HORIZONS_HOURS
    
        # Larger figure with more spacing
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.4)
    
        # SUBPLOT 1: F1 Scores
        ax1 = fig.add_subplot(gs[0, 0])
        f1_scores = [results[h]['f1'] for h in horizons]
        bars = ax1.bar(range(len(horizons)), f1_scores, color='steelblue', alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        ax1.set_xticks(range(len(horizons)))
        ax1.set_xticklabels([f'{h}h' for h in horizons], fontsize=11)
        ax1.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
        ax1.set_title('Event Detection F1 Score', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_ylim(0, max(f1_scores) * 1.2)
        
        for i, (bar, v) in enumerate(zip(bars, f1_scores)):
            ax1.text(bar.get_x() + bar.get_width()/2., v + 0.02, 
                    f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
        # SUBPLOT 2: Precision & Recall
        ax2 = fig.add_subplot(gs[0, 1])
        precision = [results[h]['precision'] for h in horizons]
        recall = [results[h]['recall'] for h in horizons]
        x_pos = np.arange(len(horizons))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, precision, width, label='Precision', 
                        color='green', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax2.bar(x_pos + width/2, recall, width, label='Recall', 
                        color='orange', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{h}h' for h in horizons], fontsize=11)
        ax2.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax2.set_title('Precision & Recall', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(fontsize=12, loc='upper right', framealpha=0.9)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_ylim(0, 1.1)
    
        # SUBPLOT 3: ROC AUC
        ax3 = fig.add_subplot(gs[0, 2])
        auc_scores = [results[h]['roc_auc'] for h in horizons]
        ax3.plot(horizons, auc_scores, marker='o', linewidth=3, markersize=12,
                color='purple', alpha=0.8, markeredgecolor='black', markeredgewidth=2)
        ax3.set_xlabel('Horizon (hours)', fontsize=13, fontweight='bold')
        ax3.set_ylabel('ROC AUC', fontsize=13, fontweight='bold')
        ax3.set_title('ROC AUC Across Horizons', fontsize=14, fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_ylim(0.4, 1.05)
        
        for x, y in zip(horizons, auc_scores):
            ax3.text(x, y + 0.02, f'{y:.3f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
        # SUBPLOT 4: Average Magnitude R² per Band (full width)
        ax4 = fig.add_subplot(gs[1, :])
        bands = self.config.FREQUENCY_BANDS
    
        avg_r2_per_band = {}
        for band in bands:
            r2_values = []
            for h in horizons:
                if 'magnitude_metrics_per_band' in results[h]:
                    mag_metrics = results[h]['magnitude_metrics_per_band'].get(band, {})
                    r2 = mag_metrics.get('r2', np.nan) if isinstance(mag_metrics, dict) else np.nan
                    if r2 != -999 and not np.isnan(r2):
                        r2_values.append(r2)
            avg_r2_per_band[band] = np.mean(r2_values) if r2_values else 0
    
        colors_map = {'approximation': 'blue', 'details_4': 'green', 'details_3': 'red',
                     'details_2': 'orange', 'details_1': 'purple'}
        colors = [colors_map.get(band, 'gray') for band in bands]
    
        bars = ax4.bar(range(len(bands)), list(avg_r2_per_band.values()),
                      color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax4.axhline(y=0, color='k', linestyle='-', linewidth=1.5, alpha=0.5)
        ax4.set_xticks(range(len(bands)))
        ax4.set_xticklabels(bands, rotation=45, ha='right', fontsize=12)
        ax4.set_ylabel('Average Magnitude R²', fontsize=13, fontweight='bold')
        ax4.set_title('Average Magnitude R² per Frequency Band (All Horizons)',
                     fontsize=15, fontweight='bold', pad=20)
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    
        for i, (bar, val) in enumerate(zip(bars, avg_r2_per_band.values())):
            y_offset = 0.03 if val > 0 else -0.05
            va = 'bottom' if val > 0 else 'top'
            ax4.text(bar.get_x() + bar.get_width()/2., val + y_offset,
                    f'{val:.3f}', ha='center', va=va, fontsize=12, fontweight='bold')
    
        # SUBPLOT 5: Reconstruction Metrics
        ax5 = fig.add_subplot(gs[2, 0])
        metrics = reconstruction_results['metrics']
        metric_names = ['R²', 'RMSE', 'MAE']
        metric_values = [metrics['r2'], metrics['rmse']/1000, metrics['mae']/1000]
        metric_colors = ['red' if metrics['r2'] < 0 else 'green', 'blue', 'orange']
    
        bars = ax5.barh(metric_names, metric_values, color=metric_colors, alpha=0.8, 
                        edgecolor='black', linewidth=1.5)
        ax5.set_xlabel('Value (RMSE/MAE in k)', fontsize=12, fontweight='bold')
        ax5.set_title('Reconstruction Metrics', fontsize=14, fontweight='bold', pad=15)
        ax5.grid(True, alpha=0.3, axis='x', linestyle='--')
    
        for i, (bar, val) in enumerate(zip(bars, [metrics['r2'], metrics['rmse'], metrics['mae']])):
            x_offset = bar.get_width() + 0.05
            ax5.text(x_offset, bar.get_y() + bar.get_height()/2.,
                    f'{val:.2f}', ha='left', va='center', fontsize=11, fontweight='bold')
    
        # SUBPLOT 6: Magnitude MAE
        ax6 = fig.add_subplot(gs[2, 1])
        mag_mae = [results[h]['magnitude_mae'] for h in horizons]
        ax6.plot(horizons, mag_mae, marker='s', linewidth=3, markersize=12,
                color='red', alpha=0.8, markeredgecolor='black', markeredgewidth=2)
        ax6.set_xlabel('Horizon (hours)', fontsize=13, fontweight='bold')
        ax6.set_ylabel('Magnitude MAE', fontsize=13, fontweight='bold')
        ax6.set_title('Magnitude Prediction Error', fontsize=14, fontweight='bold', pad=15)
        ax6.grid(True, alpha=0.3, linestyle='--')
    
        # SUBPLOT 7: Duration MAE
        ax7 = fig.add_subplot(gs[2, 2])
        dur_mae = [results[h]['duration_mae'] for h in horizons]
        ax7.plot(horizons, dur_mae, marker='D', linewidth=3, markersize=12,
                color='green', alpha=0.8, markeredgecolor='black', markeredgewidth=2)
        ax7.set_xlabel('Horizon (hours)', fontsize=13, fontweight='bold')
        ax7.set_ylabel('Duration MAE', fontsize=13, fontweight='bold')
        ax7.set_title('Duration Prediction Error', fontsize=14, fontweight='bold', pad=15)
        ax7.grid(True, alpha=0.3, linestyle='--')
    
        # SUBPLOT 8-10: Summary Statistics (bottom row)
        ax8 = fig.add_subplot(gs[3, 0])
        ax8.axis('off')
        
        avg_f1 = np.mean([results[h]['f1'] for h in horizons])
        avg_precision = np.mean([results[h]['precision'] for h in horizons])
        avg_recall = np.mean([results[h]['recall'] for h in horizons])
        
        summary_text1 = (
            "Event Detection Summary\n"
            "━" * 25 + "\n"
            f"Avg F1:        {avg_f1:.3f}\n"
            f"Avg Precision: {avg_precision:.3f}\n"
            f"Avg Recall:    {avg_recall:.3f}\n"
        )
        
        ax8.text(0.1, 0.5, summary_text1, transform=ax8.transAxes,
                fontsize=13, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1))
    
        ax9 = fig.add_subplot(gs[3, 1])
        ax9.axis('off')
        
        avg_mag_r2 = np.mean([results[h]['magnitude_r2'] for h in horizons])
        avg_mag_mae = np.mean([results[h]['magnitude_mae'] for h in horizons])
        avg_dur_mae = np.mean([results[h]['duration_mae'] for h in horizons])
        
        summary_text2 = (
            "Regression Summary\n"
            "━" * 25 + "\n"
            f"Avg Mag R²:  {avg_mag_r2:.3f}\n"
            f"Avg Mag MAE: {avg_mag_mae:.3f}\n"
            f"Avg Dur MAE: {avg_dur_mae:.3f}\n"
        )
        
        ax9.text(0.1, 0.5, summary_text2, transform=ax9.transAxes,
                fontsize=13, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=1))
    
        ax10 = fig.add_subplot(gs[3, 2])
        ax10.axis('off')
        
        summary_text3 = (
            "Overall Performance\n"
            "━" * 25 + "\n"
            f"Recon R²:    {metrics['r2']:.3f}\n"
            f"Recon RMSE:  {metrics['rmse']:.2f}\n"
            f"Horizons:    {len(horizons)}\n"
            f"Bands:       {len(bands)}\n"
        )
        
        ax10.text(0.1, 0.5, summary_text3, transform=ax10.transAxes,
                fontsize=13, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))
    
        plt.suptitle('🆕 FREQUENCY-AWARE MODEL - Performance Summary Dashboard',
                    fontsize=18, fontweight='bold', y=0.98)
    
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_events_on_timeseries(self, results: dict, original_signal: np.ndarray, 
                               events_df: pd.DataFrame, save_path: str):
        """
         NEW: Comprehensive event visualization on time-series
        
        Shows:
        - Actual time-series (background)
        - Predicted ramp events (up=red, down=green) with arrows showing direction
        - Actual ramp events (up=yellow, down=blue)
        - Predicted stationary periods (black horizontal lines)
        - Actual stationary periods (gray horizontal lines)
        - Event magnitudes and durations clearly visible
        """
        
        horizons = self.config.PREDICTION_HORIZONS_HOURS
        n_horizons = len(horizons)
        
        # Determine number of samples to show (limit to reasonable size)
        n_samples_full = len(original_signal)
        n_samples_show = min(2000, n_samples_full)
        
        fig, axes = plt.subplots(n_horizons, 1, figsize=(24, 6 * n_horizons), sharex=True)
        
        if n_horizons == 1:
            axes = [axes]
        
        for idx, h in enumerate(horizons):
            ax = axes[idx]
            
            # Get predictions and targets
            pred_occurs_prob = results[h]['predictions']['occurs'].flatten()
            pred_type = results[h]['predictions']['type']
            pred_timing = results[h]['predictions']['timing'].flatten()
            
            true_occurs = results[h]['targets']['occurs'].astype(int)
            true_type = results[h]['targets']['type'].astype(int)
            
            # Get magnitude and duration (aggregate across bands if frequency-aware)
            if 'magnitude_per_band' in results[h]['predictions']:
                # Aggregate magnitude across bands (weighted by band importance)
                band_weights = {
                    'approximation': 0.5,
                    'details_4': 0.8,
                    'details_3': 1.0,
                    'details_2': 0.8,
                    'details_1': 0.5
                }
                total_weight = sum(band_weights.values())
                
                pred_magnitude = np.zeros_like(results[h]['predictions']['magnitude_per_band']['approximation']).flatten()
                pred_duration = np.zeros_like(results[h]['predictions']['duration_per_band']['approximation']).flatten()
                
                for band, weight in band_weights.items():
                    pred_magnitude += (weight / total_weight) * results[h]['predictions']['magnitude_per_band'][band].flatten()
                    pred_duration += (weight / total_weight) * results[h]['predictions']['duration_per_band'][band].flatten()
            else:
                pred_magnitude = results[h]['predictions']['magnitude'].flatten()
                pred_duration = results[h]['predictions']['duration'].flatten()
            
            # Convert to probabilities if needed
            if pred_occurs_prob.max() > 1.0 or pred_occurs_prob.min() < 0.0:
                pred_occurs_prob = 1.0 / (1.0 + np.exp(-pred_occurs_prob))
            
            # Limit to displayable range
            n_samples = min(len(pred_occurs_prob), n_samples_show)
            x = np.arange(n_samples)
            
            signal = original_signal[:n_samples]
            
            # ========== PLOT 1: TIME-SERIES BACKGROUND ==========
            ax.plot(x, signal, color='black', linewidth=1.5, alpha=0.4, 
                   label='Original Signal', zorder=1)
            
            # Get signal range for positioning events
            signal_min = signal.min()
            signal_max = signal.max()
            signal_range = signal_max - signal_min
            signal_mid = (signal_max + signal_min) / 2
            
            # ========== PLOT 2: ACTUAL EVENTS FROM GROUND TRUTH ==========
            # Extract actual events from events_df if available
            if events_df is not None and not events_df.empty:
                actual_events_in_range = events_df[
                    (events_df['t1'] < n_samples) & (events_df['t1'] >= 0)
                ].copy()
                
                for _, event in actual_events_in_range.iterrows():
                    start = int(event['t1'])
                    end = int(min(event['t2'], n_samples))
                    
                    if end <= start or start >= n_samples:
                        continue
                    
                    event_type = event.get('event_type', 'unknown')
                    
                    if event_type == 'significant':
                        # Get magnitude and direction
                        magnitude = event.get('∆w_m', 0) if pd.notna(event.get('∆w_m')) else 0
                        angle = event.get('θ_m', 0) if pd.notna(event.get('θ_m')) else 0
                        
                        # Determine direction
                        is_up_ramp = angle > 0
                        
                        # Get signal values at event location
                        event_signal = signal[start:end] if end > start else [signal[start]]
                        y_base = np.mean(event_signal) if len(event_signal) > 0 else signal_mid
                        
                        if is_up_ramp:
                            # Actual up ramp: YELLOW/GOLD vertical span
                            ax.axvspan(start, end, alpha=0.25, color='gold', zorder=2,
                                      label='Actual Up Ramp' if idx == 0 and start == actual_events_in_range.iloc[0]['t1'] else '')
                            
                            # Arrow showing direction and magnitude
                            arrow_length = abs(magnitude) * signal_range * 0.3
                            ax.arrow(start + (end-start)/2, y_base, 0, arrow_length,
                                    head_width=(end-start)*0.2, head_length=arrow_length*0.15,
                                    fc='goldenrod', ec='darkgoldenrod', linewidth=2, 
                                    alpha=0.8, zorder=4)
                        else:
                            # Actual down ramp: BLUE/CYAN vertical span
                            ax.axvspan(start, end, alpha=0.25, color='cyan', zorder=2,
                                      label='Actual Down Ramp' if idx == 0 and start == actual_events_in_range.iloc[0]['t1'] else '')
                            
                            # Arrow showing direction and magnitude
                            arrow_length = abs(magnitude) * signal_range * 0.3
                            ax.arrow(start + (end-start)/2, y_base, 0, -arrow_length,
                                    head_width=(end-start)*0.2, head_length=arrow_length*0.15,
                                    fc='deepskyblue', ec='navy', linewidth=2,
                                    alpha=0.8, zorder=4)
                    
                    elif event_type == 'stationary':
                        # Actual stationary: GRAY horizontal line
                        y_pos = signal_mid + signal_range * 0.35
                        ax.plot([start, end], [y_pos, y_pos], 
                               color='gray', linewidth=4, alpha=0.6, solid_capstyle='butt',
                               zorder=3, label='Actual Stationary' if idx == 0 and start == actual_events_in_range.iloc[0]['t1'] else '')
            
            # ========== PLOT 3: PREDICTED EVENTS ==========
            # Classify predictions
            pred_binary = (pred_occurs_prob[:n_samples] > 0.5).astype(int)
            pred_type_class = np.argmax(pred_type[:n_samples], axis=1) if len(pred_type.shape) > 1 else pred_type[:n_samples]
            
            # Find predicted event segments
            predicted_events = []
            in_event = False
            event_start = 0
            
            for i in range(n_samples):
                if pred_binary[i] == 1 and not in_event:
                    # Event starts
                    in_event = True
                    event_start = i
                elif (pred_binary[i] == 0 or i == n_samples - 1) and in_event:
                    # Event ends
                    in_event = False
                    event_end = i
                    
                    # Get event properties
                    event_type_pred = pred_type_class[event_start]
                    event_magnitude = pred_magnitude[event_start]
                    event_duration_pred = pred_duration[event_start]
                    
                    # Ensure minimum duration
                    if event_end - event_start < 2:
                        event_end = min(event_start + int(event_duration_pred), n_samples)
                    
                    predicted_events.append({
                        'start': event_start,
                        'end': event_end,
                        'type': event_type_pred,
                        'magnitude': event_magnitude,
                        'duration': event_duration_pred
                    })
            
            # Plot predicted events
            for event in predicted_events:
                start = event['start']
                end = event['end']
                event_type = event['type']
                magnitude = event['magnitude']
                
                # Event type classification:
                # 0 = no event (shouldn't happen here)
                # 1 = significant (ramp)
                # 2 = stationary
                # 3 = both
                
                if event_type == 1 or event_type == 3:  # Significant or both
                    # Determine ramp direction from magnitude sign
                    # (assuming positive = up, negative = down)
                    is_up_ramp = magnitude >= 0
                    
                    # Get signal values at event location
                    event_signal = signal[start:end] if end > start else [signal[start]]
                    y_base = np.mean(event_signal) if len(event_signal) > 0 else signal_mid
                    
                    if is_up_ramp:
                        # Predicted up ramp: RED vertical span
                        ax.axvspan(start, end, alpha=0.2, color='red', zorder=2,
                                  edgecolor='darkred', linewidth=2, linestyle='--')
                        
                        # Arrow showing predicted direction and magnitude
                        arrow_length = abs(magnitude) * signal_range * 0.25
                        ax.arrow(start + (end-start)/2, y_base, 0, arrow_length,
                                head_width=(end-start)*0.2, head_length=arrow_length*0.15,
                                fc='red', ec='darkred', linewidth=2, alpha=0.7, zorder=4)
                    else:
                        # Predicted down ramp: GREEN vertical span
                        ax.axvspan(start, end, alpha=0.2, color='green', zorder=2,
                                  edgecolor='darkgreen', linewidth=2, linestyle='--')
                        
                        # Arrow showing predicted direction and magnitude
                        arrow_length = abs(magnitude) * signal_range * 0.25
                        ax.arrow(start + (end-start)/2, y_base, 0, -arrow_length,
                                head_width=(end-start)*0.2, head_length=arrow_length*0.15,
                                fc='green', ec='darkgreen', linewidth=2, alpha=0.7, zorder=4)
                
                if event_type == 2 or event_type == 3:  # Stationary or both
                    # Predicted stationary: BLACK horizontal line
                    y_pos = signal_mid + signal_range * 0.4
                    ax.plot([start, end], [y_pos, y_pos],
                           color='black', linewidth=5, alpha=0.7, solid_capstyle='butt',
                           zorder=3)
            
            # ========== STYLING ==========
            ax.set_ylabel('Power (MW)', fontsize=13, fontweight='bold')
            ax.set_ylim(signal_min - signal_range * 0.1, signal_max + signal_range * 0.5)
            ax.set_title(f'{h}h Horizon: Predicted vs Actual Events on Time-Series',
                        fontsize=15, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, axis='both', linestyle=':')
            
            # Custom legend (only once, avoid duplicates)
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            
            legend_elements = [
                Line2D([0], [0], color='black', linewidth=1.5, alpha=0.4, label='Original Signal'),
                Patch(facecolor='red', edgecolor='darkred', alpha=0.5, linestyle='--', label='Predicted Up Ramp'),
                Patch(facecolor='gold', alpha=0.4, label='Actual Up Ramp'),
                Patch(facecolor='green', edgecolor='darkgreen', alpha=0.5, linestyle='--', label='Predicted Down Ramp'),
                Patch(facecolor='cyan', alpha=0.4, label='Actual Down Ramp'),
                Line2D([0], [0], color='black', linewidth=5, alpha=0.7, label='Predicted Stationary'),
                Line2D([0], [0], color='gray', linewidth=4, alpha=0.6, label='Actual Stationary')
            ]
            
            ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
                     ncol=2, framealpha=0.95)
            
            # Add performance metrics box
            f1 = results[h]['f1']
            precision = results[h]['precision']
            recall = results[h]['recall']
            type_acc = results[h]['type_accuracy']
            
            n_pred_events = len(predicted_events)
            n_actual_events = len(actual_events_in_range) if events_df is not None else 0
            
            metrics_text = (
                f'Performance Metrics\n'
                f'{"─"*22}\n'
                f'F1:        {f1:.3f}\n'
                f'Precision: {precision:.3f}\n'
                f'Recall:    {recall:.3f}\n'
                f'Type Acc:  {type_acc:.3f}\n'
                f'{"─"*22}\n'
                f'Predicted: {n_pred_events}\n'
                f'Actual:    {n_actual_events}'
            )
            
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', 
                            edgecolor='black', alpha=0.9, pad=1))
        
        axes[-1].set_xlabel('Time (samples)', fontsize=13, fontweight='bold')
        
        plt.suptitle('Event Detection on Time-Series: Predicted vs Actual (Types, Directions, Durations)',
                    fontsize=18, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: events_on_timeseries.png")
    
    
    def plot_zoomed_event_comparison(self, results: dict, original_signal: np.ndarray,
                                      events_df: pd.DataFrame, save_path: str):
        """
        NEW: Zoomed view of individual events for detailed comparison
        
        Shows 6-12 selected events in detail:
        - Zoomed time-series around event
        - Predicted vs actual event boundaries
        - Event type classification
        - Magnitude and duration comparison
        """
        
        horizons = self.config.PREDICTION_HORIZONS_HOURS
        h = horizons[-1]  # Use longest horizon for most events
        
        # Get predictions
        pred_occurs_prob = results[h]['predictions']['occurs'].flatten()
        pred_type = results[h]['predictions']['type']
        
        # Convert to probabilities
        if pred_occurs_prob.max() > 1.0 or pred_occurs_prob.min() < 0.0:
            pred_occurs_prob = 1.0 / (1.0 + np.exp(-pred_occurs_prob))
        
        # Get magnitude
        if 'magnitude_per_band' in results[h]['predictions']:
            band_weights = {
                'approximation': 0.5, 'details_4': 0.8, 'details_3': 1.0,
                'details_2': 0.8, 'details_1': 0.5
            }
            total_weight = sum(band_weights.values())
            
            pred_magnitude = np.zeros_like(results[h]['predictions']['magnitude_per_band']['approximation']).flatten()
            for band, weight in band_weights.items():
                pred_magnitude += (weight / total_weight) * results[h]['predictions']['magnitude_per_band'][band].flatten()
        else:
            pred_magnitude = results[h]['predictions']['magnitude'].flatten()
        
        # Find actual events
        if events_df is None or events_df.empty:
            print("  ⚠️  No actual events available for zoomed comparison")
            return
        
        actual_events = events_df[
            (events_df['t1'] < len(pred_occurs_prob)) & (events_df['t1'] >= 0)
        ].copy()
        
        # Select 12 diverse events (different times, types)
        n_events_show = min(12, len(actual_events))
        
        if n_events_show == 0:
            print("  ⚠️  No events in range for zoomed comparison")
            return
        
        # Sample events evenly across time
        event_indices = np.linspace(0, len(actual_events)-1, n_events_show).astype(int)
        selected_events = actual_events.iloc[event_indices]
        
        # Create grid
        fig, axes = plt.subplots(3, 4, figsize=(24, 16))
        axes = axes.flatten()
        
        for plot_idx, (_, event) in enumerate(selected_events.iterrows()):
            if plot_idx >= 12:
                break
            
            ax = axes[plot_idx]
            
            # Get event location
            event_center = int((event['t1'] + event['t2']) / 2)
            window_size = 100  # Show ±100 samples around event
            
            start_window = max(0, event_center - window_size)
            end_window = min(len(original_signal), event_center + window_size)
            
            x_window = np.arange(start_window, end_window)
            signal_window = original_signal[start_window:end_window]
            
            # Plot signal
            ax.plot(x_window, signal_window, color='black', linewidth=1.5, alpha=0.6, zorder=1)
            
            # Get signal range for positioning
            signal_min = signal_window.min()
            signal_max = signal_window.max()
            signal_range = signal_max - signal_min
            signal_mid = (signal_max + signal_min) / 2
            
            # Plot actual event
            event_start = int(event['t1'])
            event_end = int(event['t2'])
            event_type = event.get('event_type', 'unknown')
            
            if event_type == 'significant':
                magnitude = event.get('∆w_m', 0) if pd.notna(event.get('∆w_m')) else 0
                angle = event.get('θ_m', 0) if pd.notna(event.get('θ_m')) else 0
                is_up = angle > 0
                
                color = 'gold' if is_up else 'cyan'
                ax.axvspan(event_start, event_end, alpha=0.3, color=color, zorder=2)
                
                # Arrow
                y_base = signal_mid
                arrow_length = abs(magnitude) * signal_range * 0.3
                arrow_dir = 1 if is_up else -1
                ax.arrow(event_center, y_base, 0, arrow_dir * arrow_length,
                        head_width=15, head_length=arrow_length*0.15,
                        fc=color, ec='black', linewidth=1.5, alpha=0.8, zorder=4)
            
            elif event_type == 'stationary':
                y_pos = signal_mid + signal_range * 0.35
                ax.plot([event_start, event_end], [y_pos, y_pos],
                       color='gray', linewidth=4, alpha=0.6, zorder=3)
            
            # Plot predicted event at this location
            if event_center < len(pred_occurs_prob):
                pred_prob = pred_occurs_prob[event_center]
                pred_mag = pred_magnitude[event_center]
                
                if pred_prob > 0.5:
                    # Predicted event exists
                    # Find predicted event boundaries
                    pred_start = event_center
                    while pred_start > 0 and pred_occurs_prob[pred_start] > 0.5:
                        pred_start -= 1
                    
                    pred_end = event_center
                    while pred_end < len(pred_occurs_prob) - 1 and pred_occurs_prob[pred_end] > 0.5:
                        pred_end += 1
                    
                    # Determine predicted type
                    is_up_pred = pred_mag >= 0
                    color_pred = 'red' if is_up_pred else 'green'
                    
                    ax.axvspan(pred_start, pred_end, alpha=0.2, color=color_pred,
                              edgecolor='black', linewidth=2, linestyle='--', zorder=2)
                    
                    # Arrow
                    y_base_pred = signal_mid - signal_range * 0.1
                    arrow_length_pred = abs(pred_mag) * signal_range * 0.25
                    arrow_dir_pred = 1 if is_up_pred else -1
                    ax.arrow(event_center, y_base_pred, 0, arrow_dir_pred * arrow_length_pred,
                            head_width=15, head_length=max(5, arrow_length_pred*0.15),
                            fc=color_pred, ec='black', linewidth=1.5, alpha=0.7, zorder=4)
            
            # Styling
            ax.set_xlim(start_window, end_window)
            ax.set_title(f'Event {plot_idx+1}: {event_type}\n'
                        f'Actual: {event_start}-{event_end} ({event_end-event_start} samples)',
                        fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add confidence/metrics text
            if event_center < len(pred_occurs_prob):
                info_text = f'Pred Conf: {pred_occurs_prob[event_center]:.2f}\nPred Mag: {pred_magnitude[event_center]:.2f}'
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for idx in range(n_events_show, 12):
            axes[idx].axis('off')
        
        plt.suptitle(f'Zoomed Event Comparison: Predicted vs Actual ({h}h Horizon)',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: zoomed_event_comparison.png")

    def plot_predicted_vs_actual_events(self, results: dict, save_path: str):
        """
        NEW: Visualize predicted events vs actual events on test time-series
        
        Shows:
        - Event occurrence predictions (probability heatmap)
        - Actual event locations (vertical lines)
        - Prediction confidence zones
        - True positives, false positives, false negatives
        """
        
        horizons = self.config.PREDICTION_HORIZONS_HOURS
        n_horizons = len(horizons)
        
        fig, axes = plt.subplots(n_horizons, 1, figsize=(20, 4 * n_horizons), sharex=True)
        
        if n_horizons == 1:
            axes = [axes]
        
        for idx, h in enumerate(horizons):
            ax = axes[idx]
            
            # Get predictions and ground truth
            pred_occurs_prob = results[h]['predictions']['occurs'].flatten()
            true_occurs = results[h]['targets']['occurs'].astype(int)
            
            # Convert logits to probabilities if needed
            if pred_occurs_prob.max() > 1.0 or pred_occurs_prob.min() < 0.0:
                pred_occurs_prob = 1.0 / (1.0 + np.exp(-pred_occurs_prob))
            
            n_samples = len(pred_occurs_prob)
            x = np.arange(n_samples)
            
            # PLOT 1: Probability heatmap background
            from matplotlib.colors import LinearSegmentedColormap
            colors = ['white', 'lightyellow', 'orange', 'red']
            n_bins = 100
            cmap = LinearSegmentedColormap.from_list('confidence', colors, N=n_bins)
            
            # Plot as colored background
            for i in range(n_samples):
                prob = pred_occurs_prob[i]
                ax.axvspan(i, i+1, alpha=prob*0.5, color=cmap(prob), zorder=0)
            
            # PLOT 2: Prediction probability line
            ax.plot(x, pred_occurs_prob, color='blue', linewidth=1.5, alpha=0.7, 
                   label='Prediction Probability', zorder=2)
            
            # PLOT 3: Threshold line
            threshold = 0.5
            ax.axhline(y=threshold, color='purple', linestyle='--', linewidth=2, 
                      alpha=0.6, label=f'Threshold ({threshold})', zorder=1)
            
            # PLOT 4: Actual event markers
            actual_event_indices = np.where(true_occurs == 1)[0]
            pred_binary = (pred_occurs_prob > threshold).astype(int)
            
            true_positives = []
            false_negatives = []
            false_positives = []
            
            for i in actual_event_indices:
                if pred_binary[i] == 1:
                    true_positives.append(i)
                else:
                    false_negatives.append(i)
            
            for i in range(n_samples):
                if pred_binary[i] == 1 and true_occurs[i] == 0:
                    false_positives.append(i)
            
            # Plot markers
            if len(true_positives) > 0:
                ax.scatter(true_positives, [1.05] * len(true_positives), 
                          marker='v', s=100, color='green', alpha=0.8, 
                          label=f'True Positive ({len(true_positives)})', zorder=5)
            
            if len(false_negatives) > 0:
                ax.scatter(false_negatives, [1.05] * len(false_negatives), 
                          marker='x', s=100, color='red', alpha=0.8, 
                          label=f'False Negative ({len(false_negatives)})', zorder=5)
            
            if len(false_positives) > 0:
                ax.scatter(false_positives, [-0.05] * len(false_positives), 
                          marker='s', s=50, color='orange', alpha=0.6, 
                          label=f'False Positive ({len(false_positives)})', zorder=4)
            
            # Vertical lines for all actual events
            for event_idx in actual_event_indices:
                ax.axvline(x=event_idx, color='black', linestyle=':', 
                          linewidth=1, alpha=0.3, zorder=1)
            
            # Styling
            ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
            ax.set_ylim(-0.1, 1.15)
            ax.set_title(f'{h}h Horizon: Predicted vs Actual Events', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10, ncol=2)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add performance metrics box
            f1 = results[h]['f1']
            precision = results[h]['precision']
            recall = results[h]['recall']
            
            metrics_text = (
                f'F1: {f1:.3f}\n'
                f'Precision: {precision:.3f}\n'
                f'Recall: {recall:.3f}\n'
                f'Events: {len(actual_event_indices)}'
            )
            
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        axes[-1].set_xlabel('Time (samples)', fontsize=12, fontweight='bold')
        
        plt.suptitle('Predicted Events vs Actual Events (Test Set)', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_reconstruction_comparison(self, reconstruction_results: dict, save_path: str):
        """Plot actual vs reconstructed time series (UNCHANGED from original)"""

        actual = reconstruction_results['actual']
        reconstructed = reconstruction_results['reconstructed']
        metrics = reconstruction_results['metrics']
        n_samples = reconstruction_results['n_samples']

        multiband_event = reconstruction_results.get('multiband_event')
        hybrid = reconstruction_results.get('hybrid')
        simple_event = reconstruction_results.get('simple_event')

        fig, axes = plt.subplots(5, 1, figsize=(20, 20))

        x = np.arange(n_samples)

        # SUBPLOT 1: All Methods Comparison
        axes[0].plot(x, actual, 'b-', linewidth=2, alpha=0.8, label='Actual')
        axes[0].plot(x, reconstructed, 'r-', linewidth=2, alpha=0.9, label='Best Reconstruction')

        if multiband_event is not None:
            axes[0].plot(x, multiband_event, 'g--', linewidth=1, alpha=0.6, label='Multi-band Event')
        if hybrid is not None:
            axes[0].plot(x, hybrid, 'c--', linewidth=1, alpha=0.6, label='Hybrid (Event+DWT)')
        if simple_event is not None:
            axes[0].plot(x, simple_event, 'm--', linewidth=1, alpha=0.6, label='Simple Event')

        axes[0].set_ylabel('Power', fontsize=12)
        axes[0].set_title('Time Series Reconstruction Comparison', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11, loc='upper right')
        axes[0].grid(True, alpha=0.3)

        metrics_text = f"RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}"
        axes[0].text(0.02, 0.98, metrics_text, transform=axes[0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # SUBPLOT 2: Absolute Error
        error = np.abs(actual - reconstructed)
        axes[1].fill_between(x, 0, error, alpha=0.5, color='red', label='Absolute Error')
        axes[1].axhline(y=metrics['mae'], color='k', linestyle='--', linewidth=2, label=f'MAE: {metrics["mae"]:.4f}')
        axes[1].set_ylabel('Absolute Error', fontsize=12)
        axes[1].set_title('Reconstruction Error', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        # SUBPLOT 3: Scatter Plot
        axes[2].scatter(actual, reconstructed, alpha=0.3, s=10, color='purple')
        min_val = min(actual.min(), reconstructed.min())
        max_val = max(actual.max(), reconstructed.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        axes[2].set_xlabel('Actual', fontsize=12)
        axes[2].set_ylabel('Reconstructed', fontsize=12)
        axes[2].set_title('Actual vs Reconstructed', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)

        # SUBPLOT 4: Error Distribution
        axes[3].hist(error, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[3].axvline(x=metrics['mae'], color='r', linestyle='--', linewidth=2, label=f'MAE: {metrics["mae"]:.4f}')
        axes[3].set_xlabel('Absolute Error', fontsize=12)
        axes[3].set_ylabel('Frequency', fontsize=12)
        axes[3].set_title('Error Distribution', fontsize=14, fontweight='bold')
        axes[3].legend(fontsize=11)
        axes[3].grid(True, alpha=0.3, axis='y')

        # SUBPLOT 5: Method Comparison
        methods = []
        r2_scores = []
        rmse_scores = []

        if multiband_event is not None:
            multiband_metrics = reconstruction_results.get('multiband_metrics', self._calculate_metrics(actual, multiband_event))
            methods.append('Multi-band\nEvent')
            r2_scores.append(multiband_metrics['r2'])
            rmse_scores.append(multiband_metrics['rmse'])

        if hybrid is not None:
            hybrid_metrics = reconstruction_results.get('hybrid_metrics', self._calculate_metrics(actual, hybrid))
            methods.append('Hybrid\n(Event+DWT)')
            r2_scores.append(hybrid_metrics['r2'])
            rmse_scores.append(hybrid_metrics['rmse'])

        if simple_event is not None:
            simple_metrics = reconstruction_results.get('simple_metrics', self._calculate_metrics(actual, simple_event))
            methods.append('Simple\nEvent')
            r2_scores.append(simple_metrics['r2'])
            rmse_scores.append(simple_metrics['rmse'])

        if len(methods) > 0:
            x_pos = np.arange(len(methods))
            width = 0.35

            bars1 = axes[4].bar(x_pos - width/2, r2_scores, width, label='R² Score', color='steelblue', alpha=0.8)
            ax4_twin = axes[4].twinx()
            bars2 = ax4_twin.bar(x_pos + width/2, rmse_scores, width, label='RMSE', color='coral', alpha=0.8)

            axes[4].set_ylabel('R² Score', fontsize=12)
            ax4_twin.set_ylabel('RMSE', fontsize=12)
            axes[4].set_xlabel('Reconstruction Method', fontsize=12)
            axes[4].set_title('Method Performance Comparison', fontsize=14, fontweight='bold')
            axes[4].set_xticks(x_pos)
            axes[4].set_xticklabels(methods, fontsize=10)
            axes[4].legend(loc='upper left', fontsize=10)
            ax4_twin.legend(loc='upper right', fontsize=10)
            axes[4].grid(True, alpha=0.3, axis='y')

            for bar in bars1:
                height = bar.get_height()
                axes[4].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            for bar in bars2:
                height = bar.get_height()
                ax4_twin.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            axes[4].text(0.5, 0.5, f'Best Method Performance\nRMSE: {metrics["rmse"]:.4f}\nR²: {metrics["r2"]:.4f}',
                        ha='center', va='center', fontsize=14, transform=axes[4].transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[4].set_title('Performance Summary', fontsize=14, fontweight='bold')
            axes[4].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_hawkes_diagnostics(model, test_loader, device, save_dir):
        """
        Visualize Hawkes Process Learning

        Shows:
        1. Learned intensity patterns
        2. Event triggering behavior
        3. Kernel decay rates
        """

        if not hasattr(model, 'hawkes_layer') or model.hawkes_layer is None:
            print("  !!!No Hawkes layer to visualize!!!")
            return

        print("   Hawkes Process Diagnostics...")

        model.eval()

        # Get one batch
        batch_X, batch_y = next(iter(test_loader))
        batch_X = batch_X.to(device)

        with torch.no_grad():
            # Get Hawkes intensity
            hawkes_features = model.hawkes_layer(batch_X)
            hawkes_intensity = hawkes_features.cpu().numpy()

        # Plot first 5 samples
        fig, axes = plt.subplots(5, 1, figsize=(14, 10))
        fig.suptitle('Hawkes Process Intensity Over Time', fontsize=16, fontweight='bold')

        for i in range(5):
            ax = axes[i]

            # Plot mean intensity across Hawkes dimensions
            intensity_mean = hawkes_intensity[i].mean(axis=1)

            ax.plot(intensity_mean, linewidth=1.5, color='orangered', label='Hawkes Intensity')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylabel('Intensity', fontsize=10)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            if i == 4:
                ax.set_xlabel('Time Step', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/hawkes_intensity_patterns.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved: hawkes_intensity_patterns.png")

        # Plot intensity distribution
        fig, ax = plt.subplots(figsize=(10, 6))

        all_intensity = hawkes_intensity.reshape(-1)
        ax.hist(all_intensity, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Hawkes Intensity Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Hawkes Intensity Values', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_intensity = all_intensity.mean()
        std_intensity = all_intensity.std()
        ax.axvline(mean_intensity, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_intensity:.3f}')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'{save_dir}/hawkes_intensity_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved: hawkes_intensity_distribution.png")
        print(f"    Intensity stats: mean={mean_intensity:.4f}, std={std_intensity:.4f}")

    def _calculate_metrics(self, actual: np.ndarray, reconstructed: np.ndarray) -> dict:
        """Calculate reconstruction metrics (UNCHANGED)"""
        mse = mean_squared_error(actual, reconstructed)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, reconstructed)

        actual_range = actual.max() - actual.min()
        nrmse = rmse / (actual_range + 1e-8)
        nmae = mae / (actual_range + 1e-8)

        r2 = r2_score(actual, reconstructed)
        mape = np.mean(np.abs((actual - reconstructed) / (actual + 1e-8))) * 100
        correlation = np.corrcoef(actual, reconstructed)[0, 1]

        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'nrmse': nrmse, 'nmae': nmae,
                'r2': r2, 'mape': mape, 'correlation': correlation}

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12: INFERENCE WORKFLOW (COMPLETE PIPELINE FOR DEPLOYMENT)
# ═══════════════════════════════════════════════════════════════════════════

import os
import pickle
import json
from datetime import datetime
from torch.utils.data import Dataset as TorchDataset

class InferenceWorkflow:
    """
    Complete inference workflow for production deployment
    
    Saves:
    1. Model weights (PyTorch + Pickle)
    2. All preprocessing artifacts (scaler, feature names, etc.)
    3. Configuration
    4. Predictions (CSV + Pickle)
    5. Execution timings
    
    Usage:
        # Save workflow
        workflow = InferenceWorkflow(config)
        workflow.save_complete_workflow(model, scaler, selected_features, ...)
        
        # Load and run inference
        workflow = InferenceWorkflow.load_workflow(checkpoint_dir)
        predictions = workflow.predict(new_data_df)
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Workflow components (populated during save/load)
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.feature_engineer = None
        self.labeler = None
        self.nominal_dict = None
        self.param_config = None
        self.dwt_decomposer = None
        self.event_extractor = None
        self.imputer = None
        self.variance_selector = None
        
        # Timings
        self.timings = {}
    
    def save_complete_workflow(self, model, scaler, selected_features, 
                               nominal_dict, param_config,
                               imputer=None, variance_selector=None,
                               results=None, reconstruction_results=None,
                               train_losses=None, val_losses=None):
        """
        Save complete workflow for inference
        
        Creates directory structure:
        results_dir/
        ├── model_checkpoints/
        │   ├── model.pt (PyTorch state dict)
        │   ├── model.pkl (Complete pickle)
        │   └── model_architecture.txt
        ├── workflow_artifacts/
        │   ├── scaler.pkl
        │   ├── selected_features.pkl
        │   ├── nominal_dict.pkl
        │   ├── param_config.pkl
        │   ├── imputer.pkl
        │   ├── variance_selector.pkl
        │   └── config.json
        ├── predictions/
        │   ├── test_predictions.csv
        │   ├── test_predictions.pkl
        │   └── reconstruction.pkl
        └── metadata/
            ├── timings.json
            ├── training_history.pkl
            └── workflow_info.txt
        """
        
        print(f"\n{'='*80}")
        print("💾 SAVING COMPLETE INFERENCE WORKFLOW")
        print(f"{'='*80}\n")
        
        import os
        
        # Create directory structure
        base_dir = self.config.RESULTS_DIR
        model_dir = self.config.get_model_save_path()
        workflow_dir = self.config.get_workflow_save_path()
        predictions_dir = self.config.get_predictions_save_path()
        metadata_dir = os.path.join(base_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        save_start = time.time()
        
        # ========== 1. SAVE MODEL ==========
        print("1️⃣ Saving model...")
        
        # PyTorch state dict (lightweight, recommended)
        model_path = os.path.join(model_dir, "model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': model.transformer.layers[0].self_attn.in_proj_weight.shape[1] if hasattr(model, 'transformer') else model.lstm.input_size,
                'hidden_size': self.config.HIDDEN_SIZE,
                'num_layers': self.config.NUM_LAYERS,
                'dropout': self.config.DROPOUT,
                'horizon_hours_list': self.config.PREDICTION_HORIZONS_HOURS,
                'short_horizon_threshold_timesteps': self.config.short_horizon_threshold,
                'frequency_bands': self.config.FREQUENCY_BANDS,
                'use_short_horizon_opt': self.config.USE_SHORT_HORIZON_OPT,
                'use_frequency_aware': self.config.USE_FREQUENCY_AWARE_PREDICTION,
                'use_hawkes': self.config.USE_HAWKES_PROCESS,
                'hawkes_dim': self.config.HAWKES_DIM,
                'hawkes_kernels': self.config.HAWKES_KERNELS,
                'hawkes_dropout': self.config.HAWKES_DROPOUT,
                'nhead': 8  # Transformer specific
            }
        }, model_path)
        print(f"   ✓ PyTorch checkpoint: {model_path}")
        
        # Full model pickle (includes architecture, heavier but standalone)
        model_pkl_path = os.path.join(model_dir, "model.pkl")
        with open(model_pkl_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"   ✓ Pickle model: {model_pkl_path}")
        
        # Model architecture as text
        arch_path = os.path.join(model_dir, "model_architecture.txt")
        with open(arch_path, 'w') as f:
            f.write(str(model))
        print(f"   ✓ Architecture: {arch_path}")
        
        # ========== 2. SAVE PREPROCESSING ARTIFACTS ==========
        print("\n2️⃣ Saving preprocessing artifacts...")
        
        artifacts = {
            'scaler': scaler,
            'selected_features': selected_features,
            'nominal_dict': nominal_dict,
            'param_config': param_config,
            'imputer': imputer,
            'variance_selector': variance_selector
        }
        
        for name, artifact in artifacts.items():
            if artifact is not None:
                artifact_path = os.path.join(workflow_dir, f"{name}.pkl")
                with open(artifact_path, 'wb') as f:
                    pickle.dump(artifact, f)
                print(f"   ✓ {name}: {artifact_path}")
        
        # ========== 3. SAVE CONFIGURATION ==========
        print("\n3️⃣ Saving configuration...")
        
        config_dict = {
            'DATA_PATH': self.config.DATA_PATH,
            'TIME_COLUMN': self.config.TIME_COLUMN,
            'TARGET_COLUMN': self.config.TARGET_COLUMN,
            'target_columns': self.config.target_columns,
            'is_multivariate': self.config.is_multivariate,
            'resolution_minutes': self.config.resolution_minutes,
            'resolution_label': self.config.resolution_label,
            'sequence_length': self.config.sequence_length,
            'prediction_horizons': self.config.prediction_horizons,
            'PREDICTION_HORIZONS_HOURS': self.config.PREDICTION_HORIZONS_HOURS,
            'WAVELET': self.config.WAVELET,
            'DWT_LEVEL': self.config.DWT_LEVEL,
            'FREQUENCY_BANDS': self.config.FREQUENCY_BANDS,
            'USE_RBA_FEATURES': self.config.USE_RBA_FEATURES,
            'USE_DWT': self.config.USE_DWT,
            'USE_SHORT_HORIZON_OPT': self.config.USE_SHORT_HORIZON_OPT,
            'USE_FREQUENCY_AWARE_PREDICTION': self.config.USE_FREQUENCY_AWARE_PREDICTION,
            'USE_HAWKES_PROCESS': self.config.USE_HAWKES_PROCESS,
            'USE_MAB_FEATURE_SELECTION': self.config.USE_MAB_FEATURE_SELECTION,
            'HIDDEN_SIZE': self.config.HIDDEN_SIZE,
            'NUM_LAYERS': self.config.NUM_LAYERS,
            'DROPOUT': self.config.DROPOUT,
            'MAX_FEATURES': self.config.MAX_FEATURES,
            'BATCH_SIZE': self.config.BATCH_SIZE,
            'saved_at': datetime.now().isoformat()
        }
        
        config_path = os.path.join(workflow_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"   ✓ Configuration: {config_path}")
        
        # ========== 4. SAVE PREDICTIONS ==========
        if results is not None:
            print("\n4️⃣ Saving predictions...")
            
            # Convert predictions to DataFrame
            predictions_df = self._predictions_to_dataframe(results)
            
            # Save as CSV
            csv_path = os.path.join(predictions_dir, "test_predictions.csv")
            predictions_df.to_csv(csv_path, index=True)
            print(f"   ✓ Predictions CSV: {csv_path}")
            
            # Save as pickle (preserves all data types)
            pkl_path = os.path.join(predictions_dir, "test_predictions.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"   ✓ Predictions pickle: {pkl_path}")
        
        # ========== 5. SAVE RECONSTRUCTION ==========
        if reconstruction_results is not None:
            print("\n5️⃣ Saving reconstruction...")
            
            recon_path = os.path.join(predictions_dir, "reconstruction.pkl")
            with open(recon_path, 'wb') as f:
                pickle.dump(reconstruction_results, f)
            print(f"   ✓ Reconstruction: {recon_path}")
        
        # ========== 6. SAVE TRAINING HISTORY ==========
        if train_losses is not None and val_losses is not None:
            print("\n6️⃣ Saving training history...")
            
            history_path = os.path.join(metadata_dir, "training_history.pkl")
            with open(history_path, 'wb') as f:
                pickle.dump({
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }, f)
            print(f"   ✓ Training history: {history_path}")
        
        # ========== 7. SAVE METADATA ==========
        print("\n7️⃣ Saving metadata...")
        
        save_time = time.time() - save_start
        self.timings['save_workflow'] = save_time
        
        timings_path = os.path.join(metadata_dir, "timings.json")
        with open(timings_path, 'w') as f:
            json.dump(self.timings, f, indent=2)
        print(f"   ✓ Timings: {timings_path}")
        
        # Workflow info
        info_path = os.path.join(metadata_dir, "workflow_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"INFERENCE WORKFLOW CHECKPOINT\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {'Hawkes+Transformer' if self.config.USE_HAWKES_PROCESS else 'Transformer'}\n")
            f.write(f"Resolution: {self.config.resolution_label}\n")
            f.write(f"Targets: {self.config.target_columns}\n")
            f.write(f"Horizons: {self.config.PREDICTION_HORIZONS_HOURS} hours\n")
            f.write(f"Features: {len(selected_features)}\n")
            f.write(f"Frequency-aware: {self.config.USE_FREQUENCY_AWARE_PREDICTION}\n")
            f.write(f"\nDirectory Structure:\n")
            f.write(f"  - model_checkpoints/: PyTorch & pickle models\n")
            f.write(f"  - workflow_artifacts/: Scaler, features, config\n")
            f.write(f"  - predictions/: Test predictions & reconstruction\n")
            f.write(f"  - metadata/: Timings & training history\n")
            f.write(f"\nUsage:\n")
            f.write(f"  workflow = InferenceWorkflow.load_workflow('{base_dir}')\n")
            f.write(f"  predictions = workflow.predict(new_data_df)\n")
        print(f"   ✓ Workflow info: {info_path}")
        
        print(f"\n{'='*80}")
        print(f"✅ WORKFLOW SAVED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"📁 Base directory: {base_dir}")
        print(f"⏱️  Save time: {save_time:.2f}s")
        print(f"{'='*80}\n")
        
        return base_dir
    
    @staticmethod
    def load_workflow(checkpoint_dir):
        """
        Load complete workflow from checkpoint
        
        Args:
            checkpoint_dir: Path to results directory containing checkpoints
        
        Returns:
            InferenceWorkflow instance ready for prediction
        """
        
        print(f"\n{'='*80}")
        print("📂 LOADING INFERENCE WORKFLOW")
        print(f"{'='*80}\n")
        
        import os
        
        load_start = time.time()
        
        # ========== 1. LOAD CONFIGURATION ==========
        print("1️⃣ Loading configuration...")
        config_path = os.path.join(checkpoint_dir, "workflow_artifacts", "config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct config object
        config = EnhancedDWTConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        print(f"   ✓ Configuration loaded")
        print(f"     Resolution: {config.resolution_label}")
        print(f"     Targets: {config.target_columns}")
        print(f"     Horizons: {config.PREDICTION_HORIZONS_HOURS}")
        
        # Create workflow instance
        workflow = InferenceWorkflow(config)
        
        # ========== 2. LOAD MODEL ==========
        print("\n2️⃣ Loading model...")
        
        model_path = os.path.join(checkpoint_dir, "model_checkpoints", "model.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=workflow.device)
        model_config = checkpoint['model_config']
        
        # Reconstruct model architecture
        workflow.model = HawkesTransformerHybrid(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            horizon_hours_list=model_config['horizon_hours_list'],
            short_horizon_threshold_timesteps=model_config['short_horizon_threshold_timesteps'],
            frequency_bands=model_config['frequency_bands'],
            use_short_horizon_opt=model_config['use_short_horizon_opt'],
            use_frequency_aware=model_config['use_frequency_aware'],
            use_hawkes=model_config['use_hawkes'],
            hawkes_dim=model_config['hawkes_dim'],
            hawkes_kernels=model_config['hawkes_kernels'],
            hawkes_dropout=model_config['hawkes_dropout'],
            nhead=model_config.get('nhead', 8)
        ).to(workflow.device)
        
        workflow.model.load_state_dict(checkpoint['model_state_dict'])
        workflow.model.eval()
        
        print(f"   ✓ Model loaded to {workflow.device}")
        
        # ========== 3. LOAD PREPROCESSING ARTIFACTS ==========
        print("\n3️⃣ Loading preprocessing artifacts...")
        
        workflow_dir = os.path.join(checkpoint_dir, "workflow_artifacts")
        
        artifacts = {
            'scaler': 'scaler.pkl',
            'selected_features': 'selected_features.pkl',
            'nominal_dict': 'nominal_dict.pkl',
            'param_config': 'param_config.pkl',
            'imputer': 'imputer.pkl',
            'variance_selector': 'variance_selector.pkl'
        }
        
        for attr_name, filename in artifacts.items():
            artifact_path = os.path.join(workflow_dir, filename)
            if os.path.exists(artifact_path):
                with open(artifact_path, 'rb') as f:
                    setattr(workflow, attr_name, pickle.load(f))
                print(f"   ✓ {attr_name}")
            else:
                print(f"   ⚠️  {attr_name} not found (optional)")
        
        # ========== 4. INITIALIZE COMPONENTS ==========
        print("\n4️⃣ Initializing workflow components...")
        
        workflow.feature_engineer = EnhancedFeatureEngineer(config)
        workflow.labeler = MultiHorizonLabeler(config)
        workflow.dwt_decomposer = DWTDecomposer(config) if config.USE_DWT else None
        workflow.event_extractor = MultiResolutionEventExtractor(config) if config.USE_DWT else None
        
        print(f"   ✓ Feature engineer")
        print(f"   ✓ Labeler")
        if workflow.dwt_decomposer:
            print(f"   ✓ DWT decomposer")
        if workflow.event_extractor:
            print(f"   ✓ Event extractor")
        
        load_time = time.time() - load_start
        workflow.timings['load_workflow'] = load_time
        
        print(f"\n{'='*80}")
        print(f"✅ WORKFLOW LOADED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"⏱️  Load time: {load_time:.2f}s")
        print(f"{'='*80}\n")
        
        return workflow
    
    def predict(self, df: pd.DataFrame, save_predictions=True):
        """
        Run complete inference pipeline on new data
        
        Args:
            df: Input dataframe with same structure as training data
            save_predictions: Whether to save predictions to disk
        
        Returns:
            Dictionary containing predictions, reconstruction, events, timings
        """
        
        print(f"\n{'='*80}")
        print("🔮 RUNNING INFERENCE PIPELINE")
        print(f"{'='*80}\n")
        
        inference_start = time.time()
        
        # Ensure time column is datetime
        if self.config.TIME_COLUMN in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[self.config.TIME_COLUMN]):
                print(f"\n⚠️  Converting {self.config.TIME_COLUMN} to datetime...")
                df = df.copy()
                df[self.config.TIME_COLUMN] = pd.to_datetime(df[self.config.TIME_COLUMN])
                print(f"   ✓ Converted to datetime")
        
        # Validate input
        print(f"Input data: {df.shape}")
        print(f"Time range: {df[self.config.TIME_COLUMN].min()} to {df[self.config.TIME_COLUMN].max()}")
        
        # ========== STEP 1: DWT DECOMPOSITION ==========
        step_start = time.time()
        print(f"\n1️⃣ DWT Decomposition...")
        
        dwt_decomposition_dict = {}
        if self.config.USE_DWT and self.dwt_decomposer:
            for target_col in self.config.target_columns:
                signal = df[target_col].values
                dwt_decomposition_dict[target_col] = self.dwt_decomposer.decompose(signal)
                print(f"   ✓ {target_col}")
        
        self.timings['dwt_decomposition'] = time.time() - step_start
        
        # ========== STEP 2: EVENT EXTRACTION ==========
        step_start = time.time()
        print(f"\n2️⃣ Event Extraction...")
        
        events_dict = {}
        event_results_dict = {}
        
        if self.config.USE_DWT and self.event_extractor:
            for target_col in self.config.target_columns:
                if target_col in dwt_decomposition_dict:
                    # Extract events per band
                    event_results = self.event_extractor.extract_events_per_band(
                        dwt_decomposition_dict[target_col],
                        self.nominal_dict[target_col],
                        self.param_config
                    )
                    
                    # Store full results (needed for reconstruction)
                    event_results_dict[target_col] = event_results
                    
                    # Get combined events
                    if 'combined' in event_results:
                        events_dict[target_col] = event_results['combined']
                    elif 'fused' in event_results:
                        events_dict[target_col] = event_results['fused']
                    else:
                        events_dict[target_col] = pd.DataFrame()
                    
                    print(f"   ✓ {target_col}: {len(events_dict[target_col])} events")
        
        self.timings['event_extraction'] = time.time() - step_start
        
        # ========== STEP 3: FEATURE ENGINEERING ==========
        step_start = time.time()
        print(f"\n3️⃣ Feature Engineering...")
        
        features = self.feature_engineer.create_features(df, events_dict, dwt_decomposition_dict)
        print(f"   ✓ Features: {features.shape}")
        
        self.timings['feature_engineering'] = time.time() - step_start
        
        # ========== STEP 4: PREPROCESSING ==========
        step_start = time.time()
        print(f"\n4️⃣ Preprocessing...")
        
        # Drop non-numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        print(f"   Numeric features: {numeric_features.shape}")
        
        # STEP 4.1: IMPUTE - HANDLE FEATURE COUNT MISMATCH
        X = numeric_features.values
        X = np.where(np.isinf(X), np.nan, X)
        
        # Check if imputer expects different number of features
        if self.imputer is not None:
            expected_features = self.imputer.n_features_in_
            actual_features = X.shape[1]
            
            if actual_features != expected_features:
                print(f"   ⚠️  Feature mismatch: have {actual_features}, need {expected_features}")
                print(f"   ⚠️  Using inference-time imputation instead of saved imputer")
                
                # Use new imputer fitted on current data
                from sklearn.impute import SimpleImputer
                temp_imputer = SimpleImputer(strategy='median')
                X = temp_imputer.fit_transform(X)
                print(f"   ✓ Imputed (fit on inference data): {X.shape}")
            else:
                # Use saved imputer
                X = self.imputer.transform(X)
                print(f"   ✓ Imputed: {X.shape}")
        else:
            from sklearn.impute import SimpleImputer
            temp_imputer = SimpleImputer(strategy='median')
            X = temp_imputer.fit_transform(X)
            print(f"   ✓ Imputed (fit on inference data): {X.shape}")
        
        # Convert back to DataFrame
        X_df = pd.DataFrame(X, columns=numeric_features.columns, index=numeric_features.index)
        
        # STEP 4.2: SELECT FEATURES (reduce to 75)
        missing_features = [f for f in self.selected_features if f not in X_df.columns]
        if missing_features:
            print(f"   ⚠️  Missing selected features: {len(missing_features)}")
            for feat in missing_features:
                X_df[feat] = 0.0
        
        X_selected = X_df[self.selected_features].values
        print(f"   ✓ Selected features: {X_selected.shape}")
        
        # STEP 4.3: SCALE
        X_scaled = self.scaler.transform(X_selected)
        print(f"   ✓ Scaled: {X_scaled.shape}")
        
        self.timings['preprocessing'] = time.time() - step_start
        
        # ========== STEP 5: CREATE SEQUENCES ==========
        step_start = time.time()
        print(f"\n5️⃣ Creating sequences...")
        
        # Create sequences manually
        sequences = []
        for i in range(len(X_scaled) - self.config.sequence_length):  # 👈 lowercase
            seq = X_scaled[i:i + self.config.sequence_length]  # 👈 lowercase
            sequences.append(seq)
        
        sequences = np.array(sequences)
        print(f"   ✓ Created {len(sequences)} sequences")
        
        # Create simple tensor dataset
        from torch.utils.data import TensorDataset, DataLoader
        
        dataset = TensorDataset(torch.FloatTensor(sequences))
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        print(f"   ✓ Sequences: {len(dataset)} samples")
        self.timings['sequence_creation'] = time.time() - step_start
        
        # ========== STEP 6: MODEL PREDICTION ==========
        step_start = time.time()
        print(f"\n6️⃣ Model prediction...")
        
        all_predictions = {h: [] for h in self.config.PREDICTION_HORIZONS_HOURS}
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                X_batch = batch[0].to(self.device)  # batch[0] instead of batch['X']
                predictions = self.model(X_batch)
                
                for h in self.config.PREDICTION_HORIZONS_HOURS:
                    pred_h = {
                        'occurs': predictions[h]['occurs'].cpu().numpy(),
                        'type': predictions[h]['type'].cpu().numpy(),
                        'timing': predictions[h]['timing'].cpu().numpy()
                    }
                    
                    if self.config.USE_FREQUENCY_AWARE_PREDICTION:
                        pred_h['magnitude_per_band'] = {
                            band: predictions[h]['magnitude_per_band'][band].cpu().numpy()
                            for band in self.config.FREQUENCY_BANDS
                        }
                        pred_h['duration_per_band'] = {
                            band: predictions[h]['duration_per_band'][band].cpu().numpy()
                            for band in self.config.FREQUENCY_BANDS
                        }
                    else:
                        pred_h['magnitude'] = predictions[h]['magnitude'].cpu().numpy()
                        pred_h['duration'] = predictions[h]['duration'].cpu().numpy()
                    
                    all_predictions[h].append(pred_h)
        
        # Concatenate batches
        for h in self.config.PREDICTION_HORIZONS_HOURS:
            all_predictions[h] = {
                key: np.concatenate([p[key] for p in all_predictions[h]], axis=0)
                if not isinstance(all_predictions[h][0][key], dict)
                else {
                    band: np.concatenate([p[key][band] for p in all_predictions[h]], axis=0)
                    for band in all_predictions[h][0][key].keys()
                }
                for key in all_predictions[h][0].keys()
            }
        
        print(f"   ✓ Predictions for {len(self.config.PREDICTION_HORIZONS_HOURS)} horizons")
        self.timings['model_prediction'] = time.time() - step_start
        
        # ========== STEP 7: RECONSTRUCTION ==========
        step_start = time.time()
        print(f"\n7️⃣ Time Series Reconstruction...")
        
        try:
            reconstructor = iDWTReconstructor(self.config)
            primary_target = self.config.target_columns[0]
            
            reconstruction_results = reconstructor.reconstruct_from_predictions(
                predictions=all_predictions,
                original_signal=df[primary_target].values,
                dwt_decomposition=dwt_decomposition_dict.get(primary_target),
                event_results_per_band=event_results_dict.get(primary_target),
                nominal=self.nominal_dict[primary_target],
                n_samples=min(self.config.RECONSTRUCT_SAMPLES, len(df))
            )
            
            print(f"   ✓ Reconstruction complete")
            print(f"     Method: {reconstruction_results.get('best_method', 'N/A')}")
            print(f"     RMSE: {reconstruction_results['metrics']['rmse']:.4f}")
            print(f"     R²: {reconstruction_results['metrics']['r2']:.4f}")
        except Exception as e:
            print(f"   ⚠️  Reconstruction failed: {e}")
            reconstruction_results = {'error': str(e)}
        
        self.timings['reconstruction'] = time.time() - step_start
        
        # ========== STEP 8: SAVE PREDICTIONS ==========
        if save_predictions:
            step_start = time.time()
            print(f"\n8️⃣ Saving predictions...")
            
            predictions_dir = self.config.get_predictions_save_path()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save predictions as CSV
            predictions_df = self._predictions_to_dataframe(all_predictions, features.index)
            csv_path = os.path.join(predictions_dir, f"predictions_{timestamp}.csv")
            predictions_df.to_csv(csv_path, index=True)
            print(f"   ✓ CSV: {csv_path}")
            
            # Save as pickle
            pkl_path = os.path.join(predictions_dir, f"predictions_{timestamp}.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump({
                    'predictions': all_predictions,
                    'reconstruction': reconstruction_results,
                    'events': events_dict,
                    'timings': self.timings
                }, f)
            print(f"   ✓ Pickle: {pkl_path}")
            
            self.timings['save_predictions'] = time.time() - step_start
        
        # ========== SUMMARY ==========
        total_time = time.time() - inference_start
        self.timings['total_inference'] = total_time
        
        print(f"\n{'='*80}")
        print(f"✅ INFERENCE COMPLETE")
        print(f"{'='*80}")
        print(f"⏱️  Timing Breakdown:")
        for step, duration in self.timings.items():
            print(f"   {step:25s}: {duration:6.2f}s")
        print(f"{'='*80}\n")
        
        return {
            'predictions': all_predictions,
            'reconstruction': reconstruction_results,
            'events': events_dict,
            'timings': self.timings
        }
    
    def _predictions_to_dataframe(self, predictions, index=None):
        """Convert predictions dict to DataFrame"""
        
        dfs = []
        
        for h in self.config.PREDICTION_HORIZONS_HOURS:
            # 👇 FIX: Handle nested structure from evaluate_model()
            if 'predictions' in predictions[h]:
                pred = predictions[h]['predictions']
            else:
                pred = predictions[h]
            
            # Base predictions
            df_h = pd.DataFrame({
                f'occurs_{h}h': pred['occurs'].flatten(),
                f'type_{h}h': np.argmax(pred['type'], axis=1) if len(pred['type'].shape) > 1 else pred['type'],
                f'timing_{h}h': pred['timing'].flatten()
            })
            
            # Frequency-aware predictions
            if self.config.USE_FREQUENCY_AWARE_PREDICTION and pred.get('magnitude_per_band'):
                for band in self.config.FREQUENCY_BANDS:
                    df_h[f'magnitude_{h}h_{band}'] = pred['magnitude_per_band'][band].flatten()
                    df_h[f'duration_{h}h_{band}'] = pred['duration_per_band'][band].flatten()
            else:
                if 'magnitude' in pred:
                    df_h[f'magnitude_{h}h'] = pred['magnitude'].flatten()
                    df_h[f'duration_{h}h'] = pred['duration'].flatten()
            
            dfs.append(df_h)
        
        result = pd.concat(dfs, axis=1)
        
        if index is not None:
            result.index = index[:len(result)]
        
        return result

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13: MAIN PIPELINE (FREQUENCY-AWARE!)
# ═══════════════════════════════════════════════════════════════════════════

class EnhancedDWTPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(config)
    
    def run_pipeline(self):
        """Execute complete pipeline"""
        
        pipeline_start = time.time()
        
        print(f"\n{'#'*80}")
        print("#" + " "*10 + " ENHANCED BASELINE B+HAWKES+DWT+MAB (UNI/MULTIVARIATE)" + " "*10 + "#")
        print(f"{'#'*80}\n")
        
        # STEP 1: Load data
        step_start = time.time()
        print(f"\n{'='*80}")
        print("STEP 1: LOAD DATA & DETECT RESOLUTION + TARGETS")
        print(f"{'='*80}\n")
        
        df = pd.read_excel(self.config.DATA_PATH)
        print(f"Loaded: {df.shape}")
        
        df[self.config.TIME_COLUMN] = pd.to_datetime(df[self.config.TIME_COLUMN])
        self.config.detect_and_set_resolution(df)
        self.config.detect_target_columns(df)
        
        # Calculate nominal for each target
        nominal_dict = {}
        for target_col in self.config.target_columns:
            nominal_dict[target_col] = df[target_col].quantile(0.95)
            print(f"Nominal ({target_col}): {nominal_dict[target_col]:.2f}")
        
        data_load_time = time.time() - step_start  # 👈 Track this
        print(f"✓ Step 1 completed in {data_load_time:.2f} seconds\n")
        
        # STEP 2: DWT per target
        step_start = time.time()
        dwt_decomposition_dict = {}
        if self.config.USE_DWT:
            print(f"\n{'='*80}")
            print("STEP 2: DWT DECOMPOSITION (PER TARGET)")
            print(f"{'='*80}\n")
            
            decomposer = DWTDecomposer(self.config)
            for target_col in self.config.target_columns:
                signal = df[target_col].values
                dwt_decomposition_dict[target_col] = decomposer.decompose(signal)
                print(f"✓ {target_col} decomposed")
        
        dwt_time = time.time() - step_start  # 👈 Track this
        print(f"✓ Step 2 completed in {dwt_time:.2f} seconds\n")
        
        # STEP 3: Extract events per target
        step_start = time.time()
        print(f"\n{'='*80}")
        print("STEP 3: EXTRACT EVENTS (PER TARGET, PER-BAND)")
        print(f"{'='*80}\n")
        
        import core.model as model
        
        events_dict = {}
        event_results_dict = {}
        param_config = None  # 👈 Initialize here
        
        for target_col in self.config.target_columns:
            print(f"\n Processing {target_col}...")
            
            normalized_data = df[target_col].values / nominal_dict[target_col]
            param_config = model.tune_mixed_strategy(df, nominal_dict[target_col])
            
            if self.config.USE_DWT and target_col in dwt_decomposition_dict:
                extractor = MultiResolutionEventExtractor(self.config)
                event_results = extractor.extract_events_per_band(
                    dwt_decomposition_dict[target_col], 
                    nominal_dict[target_col], 
                    param_config
                )
                events_dict[target_col] = event_results['combined']
                event_results_dict[target_col] = event_results
            else:
                import core.event_extraction as ee
                
                adaptive_threshold = model.calculate_adaptive_threshold(normalized_data)
                sig_threshold = adaptive_threshold * param_config.get("trad_sig_event_factor", 0.00008)
                stat_threshold = adaptive_threshold * param_config.get("trad_stat_event_factor", 0.000024)
                
                sig_events = ee.significant_events(
                    data=normalized_data, threshold=sig_threshold,
                    min_duration=param_config.get("trad_min_duration", 3),
                    min_slope=param_config.get("trad_min_slope", 0.03),
                    window_minutes=param_config.get("trad_window", 30),
                    freq_secs=param_config.get("trad_freq_secs", 800)
                )
                if not sig_events.empty:
                    sig_events['event_type'] = 'significant'
                
                stat_events = ee.stationary_events(
                    data=normalized_data, threshold=stat_threshold,
                    min_duration=param_config.get("trad_min_duration", 3),
                    min_stationary_length=param_config.get("trad_min_stationary_length", 4),
                    window_minutes=param_config.get("trad_window", 30),
                    freq_secs=param_config.get("trad_freq_secs", 800)
                )
                if not stat_events.empty:
                    stat_events['event_type'] = 'stationary'
                
                all_events = []
                if not sig_events.empty:
                    all_events.append(sig_events)
                if not stat_events.empty:
                    all_events.append(stat_events)
                
                events_dict[target_col] = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
                event_results_dict[target_col] = {'per_band': {}, 'combined': events_dict[target_col]}
            
            print(f"✓ {target_col}: {len(events_dict[target_col])} events")
        
        event_time = time.time() - step_start  # 👈 Track this
        print(f"\n✓ Step 3 completed in {event_time:.2f} seconds\n")
        
        # STEP 4: Features
        step_start = time.time()
        feature_engineer = EnhancedFeatureEngineer(self.config)
        features = feature_engineer.create_features(df, events_dict, dwt_decomposition_dict)
        
        feature_engineering_time = time.time() - step_start  # 👈 Track this
        print(f"✓ Step 4 (Feature Engineering) completed in {feature_engineering_time:.2f} seconds\n")
        
        # STEP 5: Labels (per target)
        step_start = time.time()
        labeler = MultiHorizonLabeler(self.config)
        
        # Use first target for labels
        primary_target = self.config.target_columns[0]
        
        labels = labeler.create_labels(
            features, 
            events_dict[primary_target], 
            event_results_dict[primary_target],
            dwt_decomposition=dwt_decomposition_dict.get(primary_target)
        )
        
        common_indices = features.index.intersection(labels.index)
        features = features.loc[common_indices]
        labels = labels.loc[common_indices]
        print(f"Aligned dataset: {len(features)} samples")
        
        labeling_time = time.time() - step_start  # 👈 Track this
        print(f"✓ Step 5 (Labeling) completed in {labeling_time:.2f} seconds\n")
        
        # STEP 6: MAB Feature Selection
        step_start = time.time()
        print(f"\n{'='*80}")
        print("STEP 6: HORIZON-AWARE MAB FEATURE SELECTION")
        print(f"{'='*80}\n")
        
        from sklearn.impute import SimpleImputer
        from sklearn.feature_selection import VarianceThreshold
        
        exclude_cols = [self.config.TIME_COLUMN] + self.config.target_columns + ['time_to_next_event']
        feature_cols = [col for col in features.columns 
                       if col not in exclude_cols and features[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        print(f"Engineered features: {len(feature_cols)}\n")
        
        X = features[feature_cols].values
        
        # Clean
        imputer = None  # 👈 Initialize here
        if np.isnan(X).any() or np.isinf(X).any():
            print("Cleaning NaN/Inf...")
            X = np.where(np.isinf(X), np.nan, X)
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            print("✓ Cleaned\n")
        
        # Variance filter
        print("Removing zero-variance features...")
        selector = VarianceThreshold(threshold=0.01)
        X_var = selector.fit_transform(X)
        
        selected_mask = selector.get_support()
        high_variance_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]
        print(f"✓ Kept {len(high_variance_features)} features\n")
        
        X_filtered = X_var if len(high_variance_features) > 0 else X
        if len(high_variance_features) == 0:
            high_variance_features = feature_cols
        
        # MAB with LABELS
        if self.config.USE_MAB_FEATURE_SELECTION:
            try:
                mab_selector = StratifiedMABFeatureSelector(self.config)
                X_selected, selected_features = mab_selector.select_features(
                    X_filtered, labels, high_variance_features
                )
                print("✅ MAB selection successful!\n")
                
            except Exception as e:
                print(f"\n❌ MAB failed: {e}")
                print("Falling back to Mutual Information...\n")
                
                from sklearn.feature_selection import mutual_info_classif
                y = labels[f'event_occurs_{self.config.PREDICTION_HORIZONS_HOURS[-1]}h'].values
                mi_scores = mutual_info_classif(X_filtered, y, random_state=42)
                top_k = min(self.config.MAX_FEATURES, len(high_variance_features))
                top_indices = np.argsort(mi_scores)[-top_k:]
                selected_features = [high_variance_features[i] for i in top_indices]
                X_selected = X_filtered[:, top_indices]
        else:
            print("⚠️ MAB disabled, using Mutual Information\n")
            from sklearn.feature_selection import mutual_info_classif
            y = labels[f'event_occurs_{self.config.PREDICTION_HORIZONS_HOURS[-1]}h'].values
            mi_scores = mutual_info_classif(X_filtered, y, random_state=42)
            top_k = min(self.config.MAX_FEATURES, len(high_variance_features))
            top_indices = np.argsort(mi_scores)[-top_k:]
            selected_features = [high_variance_features[i] for i in top_indices]
            X_selected = X_filtered[:, top_indices]
        
        X = X_selected
        
        if np.isnan(X).any() or np.isinf(X).any():
            X = np.where(np.isinf(X), np.nan, X)
            if imputer is None:
                imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"{'='*80}")
        print(f"FINAL FEATURE MATRIX")
        print(f"{'='*80}")
        print(f"  Shape: {X_scaled.shape}")
        print(f"  Memory: ~{X_scaled.nbytes / (1024**2):.2f} MB")
        print(f"{'='*80}\n")
        
        feature_selection_time = time.time() - step_start  # 👈 Track this
        
        # STEP 7: Split
        n_samples = len(X_scaled)
        train_size = int(n_samples * 0.70)
        val_size = int(n_samples * 0.15)
        
        X_train = X_scaled[:train_size]
        X_val = X_scaled[train_size:train_size + val_size]
        X_test = X_scaled[train_size + val_size:]
        
        y_train = labels.iloc[:train_size]
        y_val = labels.iloc[train_size:train_size + val_size]
        y_test = labels.iloc[train_size + val_size:]
        
        train_dataset = MultiTaskDataset(X_train, y_train, self.config.sequence_length, 
                                        self.config.PREDICTION_HORIZONS_HOURS, 
                                        self.config.FREQUENCY_BANDS,
                                        self.config.USE_FREQUENCY_AWARE_PREDICTION)
        val_dataset = MultiTaskDataset(X_val, y_val, self.config.sequence_length, 
                                      self.config.PREDICTION_HORIZONS_HOURS,
                                      self.config.FREQUENCY_BANDS,
                                      self.config.USE_FREQUENCY_AWARE_PREDICTION)
        test_dataset = MultiTaskDataset(X_test, y_test, self.config.sequence_length, 
                                       self.config.PREDICTION_HORIZONS_HOURS,
                                       self.config.FREQUENCY_BANDS,
                                       self.config.USE_FREQUENCY_AWARE_PREDICTION)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
    
        pos_weights = {}
        for h in self.config.PREDICTION_HORIZONS_HOURS:
            pos_rate = y_train[f'event_occurs_{h}h'].mean()
            pos_weights[h] = (1 - pos_rate) / (pos_rate + 1e-8)
            print(f"  {h}h horizon: {pos_rate*100:.2f}% events → weight={pos_weights[h]:.4f}")
        
        print(f"\n{'='*80}")
        print("MODEL INITIALIZATION")
        print(f"{'='*80}\n")
        model = HawkesTransformerHybrid(
            input_size=X_scaled.shape[1],
            hidden_size=self.config.HIDDEN_SIZE,
            num_layers=self.config.NUM_LAYERS,
            dropout=self.config.DROPOUT,
            horizon_hours_list=self.config.PREDICTION_HORIZONS_HOURS,
            short_horizon_threshold_timesteps=self.config.short_horizon_threshold,
            frequency_bands=self.config.FREQUENCY_BANDS,
            use_short_horizon_opt=self.config.USE_SHORT_HORIZON_OPT,
            use_frequency_aware=self.config.USE_FREQUENCY_AWARE_PREDICTION,
            use_hawkes=self.config.USE_HAWKES_PROCESS,
            hawkes_dim=self.config.HAWKES_DIM,
            hawkes_kernels=self.config.HAWKES_KERNELS,
            hawkes_dropout=self.config.HAWKES_DROPOUT,
            nhead=8
        ).to(self.device)
        
        print(f"✓ Model created: {'Hawkes+Transformer Hybrid' if self.config.USE_HAWKES_PROCESS else 'Transformer Baseline'}")
        print(f"  Device: {self.device}")
    
        criterion = BandWeightedLoss(
            horizons=self.config.PREDICTION_HORIZONS_HOURS,
            frequency_bands=self.config.FREQUENCY_BANDS,
            pos_weights=pos_weights,
            use_frequency_aware=self.config.USE_FREQUENCY_AWARE_PREDICTION
        ).to(self.device)
        
        print(f"\n{'='*80}")
        print("TRAINING")
        print(f"{'='*80}\n")
        train_start = time.time()
        
        if self.config.USE_STAGED_TRAINING and self.config.USE_HAWKES_PROCESS:
            model, train_losses, val_losses = train_model_staged(
                model, train_loader, val_loader, criterion, self.config, self.device
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=0.03
            )
            model, train_losses, val_losses = train_model(
                model, train_loader, val_loader, optimizer, criterion, self.config, self.device
            )
        train_time = time.time() - train_start
        
        eval_start = time.time()
        results = evaluate_model(model, test_loader, self.config, self.device)
        eval_time = time.time() - eval_start

        per_band_events = event_results_dict.get(primary_target, {}).get('per_band', {})
        all_events = []
        for band, band_events in per_band_events.items():
            if band_events is not None and not band_events.empty:
                all_events.append(band_events)
        
        events_df = pd.concat(all_events, ignore_index=True) if all_events else None
        
        print(f"\n✓ Extracted {len(events_df) if events_df is not None else 0} events for visualization")
        
        recon_start = time.time()
        print(f"\n{'='*80}")
        print("MULTI-BAND + HYBRID RECONSTRUCTION")
        print(f"{'='*80}\n")
        
        reconstructor = iDWTReconstructor(self.config)
        original_signal = df[primary_target].values
        
        reconstruction_results = reconstructor.reconstruct_from_predictions(
            predictions=results,
            original_signal=original_signal,
            dwt_decomposition=dwt_decomposition_dict.get(primary_target),
            event_results_per_band=event_results_dict.get(primary_target) if self.config.USE_DWT else None,
            nominal=nominal_dict[primary_target],
            n_samples=self.config.RECONSTRUCT_SAMPLES
        )
        
        recon_time = time.time() - recon_start
        
        viz_start = time.time()
        import os
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        
        visualizer = EnhancedDWTVisualizer(self.config)
        
        visualizer.plot_all_diagnostics(
            results=results,
            reconstruction_results=reconstruction_results,
            train_losses=train_losses,
            val_losses=val_losses,
            dwt_decomposition=dwt_decomposition_dict.get(primary_target),
            original_signal=df[primary_target].values,
            save_dir=self.config.RESULTS_DIR,
            events_df=events_df
        )
        
        viz_time = time.time() - viz_start
        
        # ========== SAVE COMPLETE WORKFLOW ==========
        print(f"\n{'='*80}")
        print("SAVING INFERENCE WORKFLOW")
        print(f"{'='*80}\n")
        
        workflow = InferenceWorkflow(self.config)
        
        # Track all timings
        workflow.timings = {
            'data_loading': data_load_time,
            'dwt_decomposition': dwt_time,
            'event_extraction': event_time,
            'feature_engineering': feature_engineering_time,
            'labeling': labeling_time,
            'feature_selection': feature_selection_time,
            'training': train_time,
            'evaluation': eval_time,
            'reconstruction': recon_time,
            'visualization': viz_time,
            'total_pipeline': time.time() - pipeline_start
        }
        
        workflow.save_complete_workflow(
            model=model,
            scaler=scaler,
            selected_features=selected_features,
            nominal_dict=nominal_dict,
            param_config=param_config,
            imputer=imputer,
            variance_selector=selector,
            results=results,
            reconstruction_results=reconstruction_results,
            train_losses=train_losses,
            val_losses=val_losses
        )
        
        if self.config.SAVE_MODELS:
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': self.config,
                'scaler': scaler,
                'selected_features': selected_features,
                'results': results,
                'reconstruction_results': reconstruction_results
            }, f"{self.config.RESULTS_DIR}/{self.config.get_ablation_name()}.pt")
        
        total_time = time.time() - pipeline_start
        
        print(f"\n{'='*80}")
        print("FINAL RESULTS (MAB + MULTIVARIATE)")
        print(f"{'='*80}\n")
        print(f" Training: {train_time:.2f}s ({train_time/60:.2f} min)")
        print(f" Total: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"\nMode: {'MULTIVARIATE' if self.config.is_multivariate else 'UNIVARIATE'}")
        print(f"Targets: {self.config.target_columns}")
        print(f"Resolution: {self.config.resolution_label}")
        print(f"Events ({primary_target}): {len(events_dict[primary_target])}")
        
        for h in self.config.PREDICTION_HORIZONS_HOURS:
            print(f"\n{h}h Horizon:")
            print(f"  F1={results[h]['f1']:.4f}, Precision={results[h]['precision']:.4f}, Recall={results[h]['recall']:.4f}")
            
            if self.config.USE_FREQUENCY_AWARE_PREDICTION and results[h].get('magnitude_metrics_per_band'):
                print(f"  Per-Band Magnitude R²:")
                for band in self.config.FREQUENCY_BANDS:
                    if band in results[h]['magnitude_metrics_per_band']:
                        mag_metrics = results[h]['magnitude_metrics_per_band'][band]
                        dur_metrics = results[h]['duration_metrics_per_band'][band]
                        
                        mag_r2 = mag_metrics.get('r2', 0.0) if isinstance(mag_metrics, dict) else 0.0
                        dur_r2 = dur_metrics.get('r2', 0.0) if isinstance(dur_metrics, dict) else 0.0
                
                        print(f"    {band:15s}: Mag={mag_r2:7.4f}, Dur={dur_r2:7.4f}")
        
        print(f"\nReconstruction:")
        print(f"  RMSE={reconstruction_results['metrics']['rmse']:.4f}, R²={reconstruction_results['metrics']['r2']:.4f}")
        print(f"\n{'='*80}\n")
        
        return results, reconstruction_results

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 14: MAIN EXECUTION (CLI-COMPATIBLE)
# ═══════════════════════════════════════════════════════════════════════════

def main(data_path=None, transfer_learning=False, config_overrides=None):
    """
    Main execution function for workflow4 (Transformer-RBA)
    
    Args:
        data_path: Path to input data (default: config.DATA_PATH)
        transfer_learning: Whether to use transfer learning mode
        config_overrides: Dict of config parameters to override
    
    Returns:
        Dictionary containing results and metadata
    """
    
    # Initialize config
    config = EnhancedDWTConfig()
    
    # Override data path if provided
    if data_path is not None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        config.DATA_PATH = data_path
        print(f"✓ Using data path: {data_path}")
    
    # Apply config overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                print(f"✓ Config override: {key} = {value}")
    
    # Transfer learning mode
    if transfer_learning:
        print(f"\n{'='*80}")
        print("🔄 TRANSFER LEARNING MODE")
        print(f"{'='*80}\n")
        
        pretrained_model_dir = config.RESULTS_DIR  # "final_experiment_transformer/"
        
        if not os.path.exists(pretrained_model_dir):
            raise FileNotFoundError(
                f"Pre-trained model not found: {pretrained_model_dir}\n"
                f"Please train a model first without --transfer flag"
            )
        
        # Load pre-trained workflow
        workflow = InferenceWorkflow.load_workflow(pretrained_model_dir)
        
        # Run inference on new data
        print(f"\n{'='*80}")
        print("RUNNING INFERENCE ON NEW DATA")
        print(f"{'='*80}\n")
        
        # Load new data
        import pandas as pd
        df_new = pd.read_excel(config.DATA_PATH)
        print(f"New data: {df_new.shape}")
        
        # Run prediction
        predictions = workflow.predict(df_new, save_predictions=True)
        
        return {
            'mode': 'transfer_learning',
            'predictions': predictions['predictions'],
            'reconstruction': predictions['reconstruction'],
            'events': predictions['events'],
            'timings': predictions['timings'],
            'model_dir': pretrained_model_dir
        }
    
    else:
        # Full training mode
        print(f"\n{'='*80}")
        print("🏋️  FULL TRAINING MODE")
        print(f"{'='*80}\n")
        
        pipeline = EnhancedDWTPipeline(config)
        results, reconstruction_results = pipeline.run_pipeline()
        
        return {
            'mode': 'full_training',
            'results': results,
            'reconstruction': reconstruction_results,
            'config': config,
            'output_dir': config.RESULTS_DIR
        }


def run_inference_only(model_dir, data_path, evaluate=True):
    """
    Run inference only (no training) using a pre-trained model
    
    Args:
        model_dir: Directory containing pre-trained model
        data_path: Path to new data
        evaluate: Whether to compute evaluation metrics (requires ground truth)
    
    Returns:
        Dictionary containing predictions and metrics
    """
    
    print(f"\n{'='*80}")
    print("🔮 INFERENCE-ONLY MODE")
    print(f"{'='*80}\n")
    
    # Validate paths
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load workflow
    workflow = InferenceWorkflow.load_workflow(model_dir)
    
    # Load new data
    import pandas as pd
    df_new = pd.read_excel(data_path)
    print(f"Input data: {df_new.shape}")
    
    # Handle time column variants
    if 'time' not in df_new.columns:
        if 'DateTime' in df_new.columns:
            df_new['time'] = df_new['DateTime']
        elif 'Time' in df_new.columns:
            df_new['time'] = df_new['Time']
    
    # Run prediction
    predictions = workflow.predict(df_new, save_predictions=True)
    
    # Convert predictions from logits to probabilities
    for h in workflow.config.PREDICTION_HORIZONS_HOURS:
        pred_occurs = predictions['predictions'][h]['occurs']
        
        # Apply sigmoid if these are logits
        if pred_occurs.max() > 1.0 or pred_occurs.min() < 0.0:
            predictions['predictions'][h]['occurs'] = 1.0 / (1.0 + np.exp(-pred_occurs))
    
    # Evaluate if requested
    metrics = None
    if evaluate:
        print(f"\n{'='*80}")
        print("COMPUTING EVALUATION METRICS")
        print(f"{'='*80}\n")
        
        metrics = _compute_inference_metrics(predictions, df_new, workflow)
    
    return {
        'predictions': predictions['predictions'],
        'reconstruction': predictions['reconstruction'],
        'events': predictions['events'],
        'metrics': metrics,
        'timings': predictions['timings']
    }


def _compute_inference_metrics(predictions, new_data, workflow):
    """
    Compute evaluation metrics for inference predictions
    
    Requires ground truth events in the new data
    """
    
    config = workflow.config
    
    # Extract ground truth events from new data
    # (You'll need to adapt this based on your data structure)
    try:
        # Assuming you have a labeler
        labeler = MultiHorizonLabeler(config)
        
        # Extract events (simplified - adjust based on your needs)
        import core.model as model
        import core.event_extraction as ee
        
        primary_target = config.target_columns[0]
        nominal = new_data[primary_target].quantile(0.95)
        normalized_data = new_data[primary_target].values / nominal
        
        param_config = model.tune_mixed_strategy(new_data, nominal)
        adaptive_threshold = model.calculate_adaptive_threshold(normalized_data)
        
        sig_threshold = adaptive_threshold * param_config.get("trad_sig_event_factor", 0.00008)
        stat_threshold = adaptive_threshold * param_config.get("trad_stat_event_factor", 0.000024)
        
        sig_events = ee.significant_events(
            data=normalized_data, threshold=sig_threshold,
            min_duration=param_config.get("trad_min_duration", 3),
            min_slope=param_config.get("trad_min_slope", 0.03),
            window_minutes=param_config.get("trad_window", 30),
            freq_secs=param_config.get("trad_freq_secs", 800)
        )
        
        stat_events = ee.stationary_events(
            data=normalized_data, threshold=stat_threshold,
            min_duration=param_config.get("trad_min_duration", 3),
            min_stationary_length=param_config.get("trad_min_stationary_length", 4),
            window_minutes=param_config.get("trad_window", 30),
            freq_secs=param_config.get("trad_freq_secs", 800)
        )
        
        all_events = []
        if not sig_events.empty:
            sig_events['event_type'] = 'significant'
            all_events.append(sig_events)
        if not stat_events.empty:
            stat_events['event_type'] = 'stationary'
            all_events.append(stat_events)
        
        events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
        
        # Create labels
        features_dummy = pd.DataFrame(index=range(len(new_data)))
        labels = labeler.create_labels(features_dummy, events_df)
        
        # Compute metrics per horizon
        metrics = {}
        
        for h in config.PREDICTION_HORIZONS_HOURS:
            pred_occurs = predictions['predictions'][h]['occurs'].flatten()
            true_occurs = labels[f'event_occurs_{h}h'].values[:len(pred_occurs)]
            
            # Binary predictions
            pred_binary = (pred_occurs > 0.5).astype(int)
            
            # Metrics
            from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
            
            f1 = f1_score(true_occurs, pred_binary, zero_division=0)
            precision = precision_score(true_occurs, pred_binary, zero_division=0)
            recall = recall_score(true_occurs, pred_binary, zero_division=0)
            
            try:
                auc = roc_auc_score(true_occurs, pred_occurs)
            except:
                auc = 0.0
            
            metrics[h] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc
            }
            
            print(f"{h}h: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, AUC={auc:.4f}")
        
        # Reconstruction metrics
        if predictions['reconstruction'] and 'metrics' in predictions['reconstruction']:
            recon_metrics = predictions['reconstruction']['metrics']
            print(f"\nReconstruction: RMSE={recon_metrics['rmse']:.4f}, R²={recon_metrics['r2']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"⚠️  Could not compute metrics: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# COMMAND-LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Transformer-RBA Workflow (Workflow4)')
    
    parser.add_argument('--data', type=str, default=None,
                       help='Path to input data (Excel file)')
    
    parser.add_argument('--transfer', action='store_true',
                       help='Use transfer learning mode (load pre-trained model)')
    
    parser.add_argument('--inference', action='store_true',
                       help='Inference-only mode (no training)')
    
    parser.add_argument('--model', type=str, default='final_experiment_transformer',
                       help='Model directory for inference/transfer learning')
    
    parser.add_argument('--config', type=str, nargs='+', default=[],
                       help='Config overrides (e.g., NUM_EPOCHS=100 BATCH_SIZE=32)')
    
    parser.add_argument('--no-evaluate', action='store_true',
                       help='Skip evaluation metrics in inference mode')
    
    args = parser.parse_args()
    
    # Parse config overrides
    config_overrides = {}
    for override in args.config:
        if '=' in override:
            key, value = override.split('=', 1)
            try:
                # Try to convert to appropriate type
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string
            
            config_overrides[key] = value
    
    # Execute based on mode
    if args.inference:
        # Inference-only mode
        result = run_inference_only(
            model_dir=args.model,
            data_path=args.data,
            evaluate=not args.no_evaluate
        )
    else:
        # Training or transfer learning mode
        result = main(
            data_path=args.data,
            transfer_learning=args.transfer,
            config_overrides=config_overrides
        )
    
    print(f"\n{'='*80}")
    print("✅ EXECUTION COMPLETED")
    print(f"{'='*80}")
    
    if result.get('mode') == 'full_training':
        print(f"Check the output directories for results.")
        print(f"Logs saved to: main_execution.log")
    elif result.get('mode') == 'transfer_learning':
        print(f"Predictions saved to: {result['model_dir']}/predictions/")
    
    print(f"\n{'='*80}\n")
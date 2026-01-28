"""
================================================================================
WORKFLOW 1: SARIMAX-RBA Integration
Traditional time series forecasting with RBA event detection
================================================================================

Architecture:
    1. RBA Event Extraction (Traditional theta algorithm)
    2. Universal Feature Engineering (works with ANY dataset)
    3. SARIMAX Time Series Forecasting
    4. Event-Based Evaluation

Key Features:
    - Automatic feature detection and engineering
    - Works with wind power, electricity prices, or any time series
    - Integrates RBA events as exogenous variables
    - Fixed ARIMA(1,0,1) with no seasonality for fast, stable training
    - Comprehensive evaluation metrics

Model Configuration:
    - ARIMA order: (1, 0, 1)
    - Seasonal order: (0, 0, 0, 0) - No seasonality
    - Expected training time: 10-20 minutes

Usage:
    # Full training
    python workflows/workflow1.py --data input_data/Baltic_Eagle.xlsx
    
    # Custom output directory
    python workflows/workflow1.py --data input_data/Baltic_Eagle.xlsx --output results/
    
    # To change model configuration, edit SARIMAXConfig class
"""

import os
import sys
import argparse
import time
import warnings
import pickle
import json
from datetime import datetime
from typing import Tuple, Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')

# Import RBA-theta modules
try:
    import core.model as model
    import core.helpers as helpers
    from core.database import RBAThetaDB
    import core.event_extraction as ee
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: RBA-theta core modules not available: {e}")
    print("   Event features will be skipped")
    CORE_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

class SARIMAXConfig:
    """Configuration for SARIMAX-RBA workflow"""
    
    def __init__(self):
        # Data configuration
        self.TIME_COLUMN = 'time'
        self.TARGET_COLUMN = None  # Auto-detect if None
        self.TURBINE_ID = 1
        
        # Split ratios
        self.TRAIN_RATIO = 0.70
        self.VAL_RATIO = 0.15
        self.TEST_RATIO = 0.15
        
        # Feature engineering
        self.MAX_FEATURES = 22
        self.MAX_LAGS = 24
        self.EXCLUDE_COLS = ['time', 'timestamp', 'date', 'datetime', 'id', 'index']
        
        # SARIMAX model configuration
        self.ARIMA_ORDER = (1, 0, 1)  # (p, d, q) - Fixed order
        self.SEASONAL_ORDER = (0, 0, 0, 0)  # No seasonality
        self.USE_SIMPLE_MODEL = False  # Deprecated - always uses fixed orders above
        
        # Training configuration
        self.OPTIMIZATION_METHOD = 'lbfgs'
        self.MAX_ITERATIONS = 50
        self.CONVERGENCE_TOL = 1e-4
        
        # Output configuration
        self.OUTPUT_DIR = 'sarimax_predictions'
        self.SAVE_PLOTS = True
        self.SAVE_RESULTS = True


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def infer_seasonal_period(index: pd.DatetimeIndex) -> int:
    """Infer seasonal period from timestamp spacing"""
    if len(index) < 3 or not hasattr(index, "inferred_type"):
        return 24  # default hourly
    
    dt = pd.Series(index).diff().mode().iloc[0]
    
    if pd.isna(dt): 
        return 24
    
    mins = max(1, int(dt / pd.Timedelta(minutes=1)))
    
    # Common granularities
    if mins == 60:     return 24    # hourly
    if mins == 30:     return 48    # 30-min
    if mins == 15:     return 96    # 15-min
    if mins == 10:     return 144   # 10-min
    if mins == 5:      return 288   # 5-min
    
    return 24


def convert_to_serializable(obj):
    """Convert numpy/pandas types to Python types for JSON export"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    return obj


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(data_path: str, config: SARIMAXConfig) -> pd.DataFrame:
    """
    Load and preprocess time series data
    
    Args:
        data_path: Path to Excel file
        config: Configuration object
    
    Returns:
        Preprocessed DataFrame with datetime index
    """
    print("\n" + "="*80)
    print("📊 LOADING DATA")
    print("="*80)
    
    # Load data
    data = pd.read_excel(data_path)
    print(f"✓ Data loaded: {data.shape}")
    print(f"  Columns: {list(data.columns)}")
    
    # Handle time column
    time_col = config.TIME_COLUMN
    if time_col not in data.columns:
        # Try common variants
        for variant in ['DateTime', 'Time', 'date', 'timestamp']:
            if variant in data.columns:
                data[time_col] = data[variant]
                break
    
    if time_col not in data.columns:
        raise KeyError(f"Time column '{time_col}' not found. Available: {list(data.columns)}")
    
    # Parse datetime
    if not np.issubdtype(data[time_col].dtype, np.datetime64):
        s = data[time_col].astype(str).str.strip()
        try:
            parsed = pd.to_datetime(s, format="ISO8601")
        except:
            try:
                parsed = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S")
            except:
                parsed = pd.to_datetime(s, format="mixed", errors="coerce")
        
        if parsed.isna().any():
            raise ValueError("Failed to parse some timestamps")
        
        data[time_col] = parsed
    
    # Set index and sort
    data = data.set_index(time_col).sort_index()
    
    # Auto-detect target column
    if config.TARGET_COLUMN is None:
        # Look for 'Power' or first numerical column
        if 'Power' in data.columns:
            config.TARGET_COLUMN = 'Power'
        else:
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                config.TARGET_COLUMN = numerical_cols[0]
            else:
                raise ValueError("No numerical column found for target")
    
    # Rename to Turbine_1 for consistency with RBA
    if config.TARGET_COLUMN != 'Turbine_1' and config.TARGET_COLUMN in data.columns:
        data = data.rename(columns={config.TARGET_COLUMN: 'Turbine_1'})
        print(f"  Renamed '{config.TARGET_COLUMN}' → 'Turbine_1'")
        config.TARGET_COLUMN = 'Turbine_1'
    
    print(f"✓ Target column: {config.TARGET_COLUMN}")
    print(f"  Time range: {data.index[0]} to {data.index[-1]}")
    print(f"  Frequency: {pd.infer_freq(data.index) or 'irregular'}")
    
    return data


# ============================================================================
# RBA EVENT EXTRACTION
# ============================================================================

def extract_rba_events(data: pd.DataFrame, nominal: float) -> Tuple[Dict, Dict]:
    """
    Extract RBA events from entire dataset using Traditional theta
    
    Args:
        data: DataFrame with power data
        nominal: Nominal power value for normalization
    
    Returns:
        Tuple of (significant_events_dict, stationary_events_dict)
    """
    if not CORE_AVAILABLE:
        print("⚠️  RBA core modules not available - skipping event extraction")
        return {}, {}
    
    print("\n" + "="*80)
    print("🔍 EXTRACTING RBA EVENTS (Traditional theta)")
    print("="*80)
    
    # Get RBA parameters
    param_config = model.tune_mixed_strategy(data, nominal)
    
    # Extract events using database approach
    with RBAThetaDB(":memory:") as db:
        db.load_data(data)
        db.normalize_data(nominal)
        turbine_ids = db.get_all_turbine_ids()
        
        all_sig_events_dict = {}
        all_stat_events_dict = {}
        
        for turbine_id in turbine_ids:
            turbine_data = db.get_turbine_data(turbine_id)
            data_values = turbine_data['normalized_value'].values
            
            # Traditional method parameters
            adaptive_threshold = model.calculate_adaptive_threshold(data_values)
            trad_sig_factor = param_config.get("trad_sig_event_factor", 0.00003)
            trad_stat_factor = param_config.get("trad_stat_event_factor", 0.00009)
            
            # Extract significant events
            all_sig_events_dict[turbine_id] = ee.significant_events(
                data=data_values,
                threshold=adaptive_threshold * trad_sig_factor,
                min_duration=param_config.get("trad_min_duration", 3),
                min_slope=param_config.get("trad_min_slope", 0.05),
                window_minutes=param_config.get("trad_window", 60),
                freq_secs=param_config.get("trad_freq_secs", 100),
            )
            
            # Extract stationary events
            all_stat_events_dict[turbine_id] = ee.stationary_events(
                data=data_values,
                threshold=adaptive_threshold * trad_stat_factor,
                min_duration=param_config.get("trad_min_duration", 3),
                min_stationary_length=param_config.get("trad_min_stationary_length", 7),
                window_minutes=param_config.get("trad_window", 60),
                freq_secs=param_config.get("trad_freq_secs", 100),
            )
    
    print(f"✓ Events extracted for {len(turbine_ids)} turbine(s)")
    
    return all_sig_events_dict, all_stat_events_dict


def convert_events_to_dataframe(sig_dict: Dict, stat_dict: Dict) -> pd.DataFrame:
    """Convert event dictionaries to combined DataFrame"""
    all_events = []
    
    for turbine_id in sig_dict.keys():
        # Significant events
        sig_events = sig_dict[turbine_id]
        if not sig_events.empty:
            sig_copy = sig_events.copy()
            sig_copy['event_type'] = 'significant'
            turbine_num = int(turbine_id.split('_')[-1]) if '_' in str(turbine_id) else 1
            sig_copy['turbine_id'] = turbine_num
            all_events.append(sig_copy)
        
        # Stationary events
        stat_events = stat_dict[turbine_id]
        if not stat_events.empty:
            stat_copy = stat_events.copy()
            stat_copy['event_type'] = 'stationary'
            turbine_num = int(turbine_id.split('_')[-1]) if '_' in str(turbine_id) else 1
            stat_copy['turbine_id'] = turbine_num
            all_events.append(stat_copy)
    
    if all_events:
        result = pd.concat(all_events, ignore_index=True)
        print(f"✓ Total events: {len(result)}")
        if not result.empty:
            print("  Event types:")
            print(result['event_type'].value_counts().to_dict())
        return result
    else:
        return pd.DataFrame()


# ============================================================================
# DATA SPLITTING
# ============================================================================

def split_data_and_events(data: pd.DataFrame, events_df: pd.DataFrame, 
                          config: SARIMAXConfig) -> Dict:
    """
    Split data and events into train/val/test maintaining alignment
    
    Args:
        data: Full dataset
        events_df: Full events DataFrame
        config: Configuration object
    
    Returns:
        Dictionary with split data and events
    """
    print("\n" + "="*80)
    print("✂️  SPLITTING DATA")
    print("="*80)
    
    n = len(data)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))
    
    # Split raw data
    train_raw = data.iloc[:train_end].copy()
    val_raw = data.iloc[train_end:val_end].copy()
    test_raw = data.iloc[val_end:].copy()
    
    # Filter events by time period
    def filter_events(events, start_idx, end_idx):
        if events.empty:
            return events.copy()
        
        period_events = events[
            (events['t1'] >= start_idx) & 
            (events['t2'] < end_idx)
        ].copy()
        
        if not period_events.empty:
            period_events['t1'] = period_events['t1'] - start_idx
            period_events['t2'] = period_events['t2'] - start_idx
        
        return period_events
    
    train_events = filter_events(events_df, 0, train_end)
    val_events = filter_events(events_df, train_end, val_end)
    test_events = filter_events(events_df, val_end, n)
    
    print(f"✓ Data split:")
    print(f"  Train: {train_raw.shape[0]} samples, {len(train_events)} events")
    print(f"  Val:   {val_raw.shape[0]} samples, {len(val_events)} events")
    print(f"  Test:  {test_raw.shape[0]} samples, {len(test_events)} events")
    
    return {
        'train_raw': train_raw,
        'val_raw': val_raw,
        'test_raw': test_raw,
        'train_events': train_events,
        'val_events': val_events,
        'test_events': test_events,
        'full_data': data
    }


# ============================================================================
# UNIVERSAL FEATURE ENGINEERING
# ============================================================================

def create_universal_features(raw_data: pd.DataFrame, events_data: pd.DataFrame,
                              original_data: pd.DataFrame, period_name: str,
                              config: SARIMAXConfig) -> pd.DataFrame:
    """
    FULLY UNIVERSAL feature engineering for SARIMAX
    Works with ANY dataset - automatically detects and engineers ALL features
    
    Args:
        raw_data: DataFrame with target column
        events_data: DataFrame with RBA event annotations (can be empty)
        original_data: DataFrame with ANY features from ANY domain
        period_name: String label (e.g., "Training", "Validation", "Test")
        config: Configuration object
    
    Returns:
        DataFrame with engineered features
    """
    features = pd.DataFrame(index=raw_data.index)
    target_col = config.TARGET_COLUMN
    
    print(f"\n{'='*80}")
    print(f"🔧 UNIVERSAL FEATURE ENGINEERING - {period_name.upper()}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # 1. TARGET VARIABLE
    # ========================================================================
    features[target_col] = raw_data[target_col]
    print(f"✓ Target variable: '{target_col}'")
    print(f"  Range: {features[target_col].min():.4f} to {features[target_col].max():.4f}")
    
    # ========================================================================
    # 2. AUTO-DETECT FEATURES FROM ORIGINAL DATA
    # ========================================================================
    available_cols = [col for col in original_data.columns 
                     if col not in config.EXCLUDE_COLS 
                     and col != target_col]
    
    print(f"\n📊 Found {len(available_cols)} potential features")
    
    # Classify features
    numerical_features = []
    categorical_features = []
    circular_features = []
    binary_features = []
    
    for col in available_cols:
        try:
            col_data = original_data[col]
            
            if col_data.isna().all():
                continue
            
            if pd.api.types.is_numeric_dtype(col_data):
                unique_vals = col_data.nunique()
                
                if unique_vals == 2:
                    binary_features.append(col)
                elif any(x in col.lower() for x in ['direction', 'angle', 'azimuth', 'bearing', 'degree']):
                    circular_features.append(col)
                else:
                    numerical_features.append(col)
            else:
                categorical_features.append(col)
        
        except Exception:
            pass
    
    # ========================================================================
    # 3. PROCESS NUMERICAL FEATURES
    # ========================================================================
    if numerical_features:
        print(f"\n📊 Processing {len(numerical_features)} numerical features...")
        
        for col in numerical_features:
            try:
                base_name = col.replace(' ', '_').replace('-', '_')
                features[base_name] = original_data.loc[raw_data.index, col]
                
                # Lags
                for lag in [1, 2, 3, 6, 12, 24]:
                    if lag <= len(features) // 10:
                        features[f'{base_name}_lag{lag}'] = features[base_name].shift(lag)
                
                # Differences
                features[f'{base_name}_diff1'] = features[base_name].diff(1)
                
                # Rolling statistics
                for window in [6, 12, 24]:
                    if window <= len(features) // 5:
                        features[f'{base_name}_rolling_mean_{window}'] = features[base_name].rolling(window).mean()
                        features[f'{base_name}_rolling_std_{window}'] = features[base_name].rolling(window).std()
                
                # Non-linear
                if (features[base_name] >= 0).all():
                    features[f'{base_name}_squared'] = features[base_name] ** 2
                    features[f'{base_name}_cubed'] = features[base_name] ** 3
            
            except Exception as e:
                print(f"    ⚠️  Error processing {col}: {e}")
    
    # ========================================================================
    # 4. PROCESS CIRCULAR FEATURES
    # ========================================================================
    if circular_features:
        print(f"\n🔄 Processing {len(circular_features)} circular features...")
        
        for col in circular_features:
            try:
                base_name = col.replace(' ', '_').replace('-', '_')
                features[base_name] = original_data.loc[raw_data.index, col]
                
                # Circular encoding
                features[f'{base_name}_sin'] = np.sin(np.radians(features[base_name]))
                features[f'{base_name}_cos'] = np.cos(np.radians(features[base_name]))
                
                # Circular difference
                diff = features[base_name].diff()
                diff = ((diff + 180) % 360) - 180
                features[f'{base_name}_change'] = diff
            
            except Exception as e:
                print(f"    ⚠️  Error processing {col}: {e}")
    
    # ========================================================================
    # 5. TARGET VARIABLE ENGINEERING
    # ========================================================================
    print(f"\n🎯 Engineering target variable features...")
    
    # Lags
    for lag in [1, 2, 3, 6, 12, 24]:
        if lag <= len(features) // 10:
            features[f'{target_col}_lag{lag}'] = features[target_col].shift(lag)
    
    # Differences
    features[f'{target_col}_diff1'] = features[target_col].diff(1)
    features[f'{target_col}_diff2'] = features[target_col].diff(2)
    
    # Rolling statistics
    for window in [6, 12, 24, 168]:
        if window <= len(features) // 5:
            features[f'{target_col}_rolling_mean_{window}'] = features[target_col].rolling(window).mean()
            features[f'{target_col}_rolling_std_{window}'] = features[target_col].rolling(window).std()
            features[f'{target_col}_rolling_min_{window}'] = features[target_col].rolling(window).min()
            features[f'{target_col}_rolling_max_{window}'] = features[target_col].rolling(window).max()
    
    # ========================================================================
    # 6. RBA EVENT FEATURES
    # ========================================================================
    if not events_data.empty and CORE_AVAILABLE:
        print(f"\n🎯 Processing RBA event features...")
        
        event_magnitude = np.zeros(len(raw_data))
        event_duration = np.zeros(len(raw_data))
        event_slope = np.zeros(len(raw_data))
        event_sigma = np.zeros(len(raw_data))
        event_type_sig = np.zeros(len(raw_data))
        event_type_stat = np.zeros(len(raw_data))
        time_since_last_event = np.zeros(len(raw_data))
        event_intensity = np.zeros(len(raw_data))
        
        turbine_events = events_data[events_data['turbine_id'] == config.TURBINE_ID]
        if turbine_events.empty:
            turbine_events = events_data  # Use all events if filtered is empty
        
        last_event_end = 0
        
        for _, event in turbine_events.iterrows():
            try:
                start_idx = int(event['t1'])
                end_idx = int(event['t2'])
                
                if 0 <= start_idx < len(raw_data) and 0 <= end_idx < len(raw_data):
                    duration = end_idx - start_idx + 1
                    
                    if event['event_type'] == 'significant':
                        magnitude = abs(event.get('∆w_m', 0))
                        slope = event.get('θ_m', 0)
                        sigma = event.get('σ_m', 0)
                        
                        event_magnitude[start_idx:end_idx+1] = magnitude
                        event_slope[start_idx:end_idx+1] = slope
                        event_sigma[start_idx:end_idx+1] = sigma
                        event_type_sig[start_idx:end_idx+1] = 1
                        event_intensity[start_idx:end_idx+1] = magnitude / max(duration, 1)
                    
                    elif event['event_type'] == 'stationary':
                        sigma = event.get('σ_s', 0)
                        event_sigma[start_idx:end_idx+1] = sigma
                        event_type_stat[start_idx:end_idx+1] = 1
                        event_intensity[start_idx:end_idx+1] = sigma
                    
                    event_duration[start_idx:end_idx+1] = duration
                    
                    if last_event_end > 0:
                        gap = start_idx - last_event_end
                        time_since_last_event[start_idx:end_idx+1] = gap
                    
                    last_event_end = end_idx
            
            except Exception:
                continue
        
        # Add event features
        features['event_magnitude'] = event_magnitude
        features['event_duration'] = event_duration
        features['event_slope'] = event_slope
        features['event_sigma'] = event_sigma
        features['is_significant_event'] = event_type_sig
        features['is_stationary_event'] = event_type_stat
        features['time_since_last_event'] = time_since_last_event
        features['event_intensity'] = event_intensity
        
        # Event rolling features
        features['event_mag_ma3'] = pd.Series(event_magnitude).rolling(3).mean().fillna(0).values
        features['event_mag_ma12'] = pd.Series(event_magnitude).rolling(12).mean().fillna(0).values
        features['event_freq_24h'] = pd.Series(event_type_sig + event_type_stat).rolling(24).sum().fillna(0).values
        
        non_zero_events = (event_magnitude > 0).sum()
        print(f"  ✓ Event coverage: {non_zero_events} time points ({non_zero_events/len(features)*100:.1f}%)")
    
    # ========================================================================
    # 7. TIME-BASED FEATURES
    # ========================================================================
    if isinstance(features.index, pd.DatetimeIndex):
        print(f"\n📅 Creating time-based features...")
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['day_of_month'] = features.index.day
        features['month'] = features.index.month
        features['quarter'] = features.index.quarter
        features['is_weekend'] = (features.index.dayofweek >= 5).astype(int)
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    
    # ========================================================================
    # 8. FILL MISSING VALUES
    # ========================================================================
    features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    # ========================================================================
    # 9. SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"FEATURE ENGINEERING SUMMARY - {period_name.upper()}")
    print(f"{'='*80}")
    print(f"Final shape: {features.shape[0]} rows × {features.shape[1]} columns\n")
    
    target_features = [col for col in features.columns if target_col in col]
    event_features = [col for col in features.columns if 'event' in col.lower()]
    time_features = [col for col in features.columns if any(x in col for x in ['hour', 'day', 'month', 'weekend', 'quarter'])]
    original_features = [col for col in features.columns if col not in target_features and col not in event_features and col not in time_features]
    
    print(f"✓ Target-derived features: {len(target_features)}")
    print(f"✓ Original dataset features: {len(original_features)}")
    print(f"✓ Event features: {len(event_features)}")
    print(f"✓ Time-based features: {len(time_features)}")
    print(f"\n✓ TOTAL FEATURES: {features.shape[1]}")
    print(f"{'='*80}\n")
    
    return features


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_exog_features(features_df: pd.DataFrame, target_col: str,
                         max_features: int = 30) -> List[str]:
    """
    Intelligently select exogenous features for SARIMAX
    Prioritizes: lags, events, important features
    
    Args:
        features_df: DataFrame with all engineered features
        target_col: Name of target column
        max_features: Maximum number of exogenous features
    
    Returns:
        List of selected column names
    """
    selected = []
    
    # Priority 1: Target Lags
    lag_cols = [col for col in features_df.columns if target_col in col and 'lag' in col]
    lag_cols_sorted = sorted(lag_cols, key=lambda x: int(x.split('lag')[-1]) if 'lag' in x else 999)
    selected.extend(lag_cols_sorted[:6])
    
    # Priority 2: Event Features
    event_priority = [
        'event_magnitude',
        'event_intensity',
        'is_significant_event',
        'is_stationary_event',
        'event_duration',
        'event_slope',
        'time_since_last_event',
        'event_freq_24h',
        'event_mag_ma3'
    ]
    event_cols = [col for col in event_priority if col in features_df.columns]
    selected.extend(event_cols)
    
    # Priority 3: Wind Features (if available)
    wind_speed_priority = [
        'Windspeed',
        'Windspeed_cubed',
        'Windspeed_rolling_mean_6',
        'Windspeed_rolling_std_6'
    ]
    wind_speed_cols = [col for col in wind_speed_priority if col in features_df.columns]
    selected.extend(wind_speed_cols)
    
    wind_dir_priority = [
        'Wind_Direction_sin',
        'Wind_Direction_cos',
        'Wind_Direction_change'
    ]
    wind_dir_cols = [col for col in wind_dir_priority if col in features_df.columns]
    selected.extend(wind_dir_cols)
    
    # Priority 4: Other features up to limit
    remaining_budget = max_features - len(selected)
    
    if remaining_budget > 0:
        other_numerical = [col for col in features_df.columns 
                          if col not in selected 
                          and col != target_col
                          and features_df[col].dtype in ['float64', 'int64']
                          and not any(x in col for x in ['hour', 'day', 'month', 'weekend'])]
        
        priority_patterns = ['rolling_mean', 'rolling_std', 'normalized', '_x_', 'during']
        other_priority = []
        
        for pattern in priority_patterns:
            other_priority.extend([col for col in other_numerical if pattern in col and col not in other_priority])
        
        other_priority.extend([col for col in other_numerical if col not in other_priority])
        selected.extend(other_priority[:remaining_budget])
    
    # Ensure all selected columns exist
    selected = [col for col in selected if col in features_df.columns]
    
    print(f"\n📋 Feature Selection:")
    print(f"   Total features selected: {len(selected)}")
    print(f"   Sample features: {selected[:10]}")
    
    return selected


# ============================================================================
# SARIMAX MODEL TRAINING
# ============================================================================

def build_endog_exog(df: pd.DataFrame, target_col: str, exog_cols: List[str],
                     scaler: Optional[StandardScaler] = None,
                     fit_scaler: bool = True) -> Tuple:
    """
    Prepare endogenous and exogenous variables for SARIMAX
    
    Returns:
        (y, X, scaler)
    """
    y = df[target_col].astype("float64")
    X = df[exog_cols].astype("float64")
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit_scaler:
        Xs = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    else:
        Xs = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    
    # Combine and clean
    Z = pd.concat([y, Xs], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    
    return Z[target_col], Z[Xs.columns], scaler


def train_sarimax_model(y_train: pd.Series, X_train: pd.DataFrame,
                        config: SARIMAXConfig, seasonal_m: int):
    """
    Train SARIMAX model
    
    Returns:
        Fitted SARIMAX results
    """
    print("\n" + "="*80)
    print("🎓 TRAINING SARIMAX MODEL")
    print("="*80)
    
    # Use fixed model orders from config
    model_order = config.ARIMA_ORDER
    model_seasonal = config.SEASONAL_ORDER
    
    print(f"\n⚙️  SARIMAX Configuration:")
    print(f"   ARIMA order: {model_order}")
    print(f"   Seasonal order: {model_seasonal}")
    print(f"   Training samples: {len(y_train):,}")
    print(f"   Exogenous features: {X_train.shape[1]}")
    
    # Build model
    model = SARIMAX(
        endog=y_train,
        exog=X_train,
        order=model_order,
        seasonal_order=model_seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
        trend='c',
        simple_differencing=True,
        hamilton_representation=False
    )
    
    print("\n🔧 Training model...")
    start_time = time.time()
    
    results = model.fit(
        disp=True,
        maxiter=config.MAX_ITERATIONS,
        method=config.OPTIMIZATION_METHOD,
        ftol=config.CONVERGENCE_TOL,
        gtol=config.CONVERGENCE_TOL
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Training complete in {elapsed_time/60:.2f} minutes!")
    print(f"\n📊 Model Summary:")
    print(f"   AIC: {results.aic:.2f}")
    print(f"   BIC: {results.bic:.2f}")
    print(f"   Log-Likelihood: {results.llf:.2f}")
    print(f"   Parameters: {len(results.params)}")
    print(f"   Converged: {results.mle_retvals.get('converged', False)}")
    
    return results


# ============================================================================
# PREDICTION AND EVALUATION
# ============================================================================

def make_predictions(results, y_train, y_val, y_test, 
                     X_train, X_val, X_test) -> Dict:
    """
    Generate predictions for all sets
    
    Returns:
        Dictionary with predictions and metrics
    """
    print("\n" + "="*80)
    print("🔮 MAKING PREDICTIONS")
    print("="*80)
    
    # Training predictions
    print("\n1️⃣ Training predictions...")
    y_train_pred = results.fittedvalues
    
    # Validation predictions
    print("2️⃣ Validation predictions...")
    y_val_pred = results.forecast(steps=len(y_val), exog=X_val)
    
    # Test predictions
    print("3️⃣ Test predictions...")
    results_extended = results.append(y_val, exog=X_val, refit=False)
    y_test_pred = results_extended.forecast(steps=len(y_test), exog=X_test)
    
    # Calculate metrics
    metrics = {}
    for name, y_true, y_pred in [
        ('train', y_train, y_train_pred),
        ('val', y_val, y_val_pred),
        ('test', y_test, y_test_pred)
    ]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        metrics[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
    
    print("\n" + "="*80)
    print("📊 PERFORMANCE SUMMARY")
    print("="*80)
    
    for name in ['train', 'val', 'test']:
        m = metrics[name]
        print(f"\n{name.upper()} Set:")
        print(f"   MAE:  {m['MAE']:.4f}")
        print(f"   RMSE: {m['RMSE']:.4f}")
        print(f"   R²:   {m['R2']:.4f}")
    
    return {
        'predictions': {
            'train': {'actual': y_train, 'predicted': y_train_pred},
            'val': {'actual': y_val, 'predicted': y_val_pred},
            'test': {'actual': y_test, 'predicted': y_test_pred}
        },
        'metrics': metrics
    }


# ============================================================================
# POST-HOC EVENT EXTRACTION AND EVALUATION
# ============================================================================

def extract_events_from_predictions(predicted_series: pd.Series, nominal: float,
                                    turbine_id: str = 'Turbine_1') -> pd.DataFrame:
    """
    Extract RBA events from predicted time series
    
    Args:
        predicted_series: Predicted power values
        nominal: Nominal power for normalization
        turbine_id: Turbine identifier
    
    Returns:
        DataFrame with predicted events
    """
    if not CORE_AVAILABLE:
        print("⚠️  RBA core modules not available - skipping event extraction")
        return pd.DataFrame()
    
    print("\n⚙️  Extracting events from predicted time series...")
    
    # Create DataFrame in format expected by RBA
    pred_df = pd.DataFrame({
        turbine_id: predicted_series.values
    }, index=predicted_series.index)
    
    # Get RBA parameters
    param_config = model.tune_mixed_strategy(pred_df, nominal)
    
    # Extract events
    with RBAThetaDB(":memory:") as db:
        db.load_data(pred_df)
        db.normalize_data(nominal)
        
        turbine_data = db.get_turbine_data(turbine_id)
        data_values = turbine_data['normalized_value'].values
        
        # Traditional method parameters
        adaptive_threshold = model.calculate_adaptive_threshold(data_values)
        trad_sig_factor = param_config.get("trad_sig_event_factor", 0.00003)
        trad_stat_factor = param_config.get("trad_stat_event_factor", 0.00009)
        
        # Extract significant events
        sig_events = ee.significant_events(
            data=data_values,
            threshold=adaptive_threshold * trad_sig_factor,
            min_duration=param_config.get("trad_min_duration", 3),
            min_slope=param_config.get("trad_min_slope", 0.05),
            window_minutes=param_config.get("trad_window", 60),
            freq_secs=param_config.get("trad_freq_secs", 100),
        )
        
        # Extract stationary events
        stat_events = ee.stationary_events(
            data=data_values,
            threshold=adaptive_threshold * trad_stat_factor,
            min_duration=param_config.get("trad_min_duration", 3),
            min_stationary_length=param_config.get("trad_min_stationary_length", 7),
            window_minutes=param_config.get("trad_window", 60),
            freq_secs=param_config.get("trad_freq_secs", 100),
        )
    
    # Convert to combined DataFrame
    all_events = []
    
    if not sig_events.empty:
        sig_copy = sig_events.copy()
        sig_copy['event_type'] = 'significant'
        sig_copy['turbine_id'] = 1
        all_events.append(sig_copy)
    
    if not stat_events.empty:
        stat_copy = stat_events.copy()
        stat_copy['event_type'] = 'stationary'
        stat_copy['turbine_id'] = 1
        all_events.append(stat_copy)
    
    if all_events:
        result = pd.concat(all_events, ignore_index=True)
        print(f"   ✓ Predicted events extracted: {len(result)}")
        if not result.empty:
            print(f"      Event types: {result['event_type'].value_counts().to_dict()}")
        return result
    else:
        return pd.DataFrame()


def calculate_event_overlap_metrics(actual_events: pd.DataFrame, 
                                    predicted_events: pd.DataFrame,
                                    tolerance: int = 5) -> Dict:
    """
    Calculate precision, recall, F1 for event detection using temporal overlap
    
    Args:
        actual_events: Ground truth events
        predicted_events: Events extracted from predictions
        tolerance: Time points tolerance for matching
    
    Returns:
        Dictionary with metrics
    """
    if actual_events.empty or predicted_events.empty:
        return {
            'true_positives': 0,
            'false_positives': len(predicted_events),
            'false_negatives': len(actual_events),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    true_positives = 0
    matched_predicted = set()
    matched_actual = set()
    
    # For each actual event, check if any predicted event overlaps
    for idx_actual, actual_event in actual_events.iterrows():
        actual_start = actual_event['t1']
        actual_end = actual_event['t2']
        actual_type = actual_event['event_type']
        
        for idx_pred, pred_event in predicted_events.iterrows():
            if idx_pred in matched_predicted:
                continue
            
            pred_start = pred_event['t1']
            pred_end = pred_event['t2']
            pred_type = pred_event['event_type']
            
            # Check for temporal overlap with tolerance
            overlap_start = max(actual_start - tolerance, pred_start)
            overlap_end = min(actual_end + tolerance, pred_end)
            
            # Check if type matches and there's overlap
            if pred_type == actual_type and overlap_start <= overlap_end:
                true_positives += 1
                matched_predicted.add(idx_pred)
                matched_actual.add(idx_actual)
                break
    
    false_positives = len(predicted_events) - len(matched_predicted)
    false_negatives = len(actual_events) - len(matched_actual)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matched_actual': len(matched_actual),
        'matched_predicted': len(matched_predicted)
    }


def evaluate_event_detection(actual_events: pd.DataFrame,
                             predicted_events: pd.DataFrame,
                             tolerance: int = 5) -> Dict:
    """
    Comprehensive event detection evaluation
    
    Returns:
        Dictionary with overall and per-type metrics
    """
    print("\n" + "="*80)
    print("🎯 EVENT DETECTION EVALUATION")
    print("="*80)
    
    # Overall metrics
    overall_metrics = calculate_event_overlap_metrics(actual_events, predicted_events, tolerance)
    
    print(f"\n📊 Overall Event Detection:")
    print(f"   True Positives:  {overall_metrics['true_positives']}")
    print(f"   False Positives: {overall_metrics['false_positives']}")
    print(f"   False Negatives: {overall_metrics['false_negatives']}")
    print(f"   Precision: {overall_metrics['precision']:.4f}")
    print(f"   Recall:    {overall_metrics['recall']:.4f}")
    print(f"   F1-Score:  {overall_metrics['f1']:.4f}")
    
    # Per-type metrics
    by_type_metrics = {}
    
    for event_type in ['significant', 'stationary']:
        actual_type = actual_events[actual_events['event_type'] == event_type]
        pred_type = predicted_events[predicted_events['event_type'] == event_type]
        
        if len(actual_type) > 0:
            type_metrics = calculate_event_overlap_metrics(actual_type, pred_type, tolerance)
            by_type_metrics[event_type] = type_metrics
            
            print(f"\n   {event_type.capitalize()} Events:")
            print(f"      Actual: {len(actual_type)}, Predicted: {len(pred_type)}")
            print(f"      Precision: {type_metrics['precision']:.4f}")
            print(f"      Recall:    {type_metrics['recall']:.4f}")
            print(f"      F1-Score:  {type_metrics['f1']:.4f}")
    
    return {
        'overall': overall_metrics,
        'by_type': by_type_metrics
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(predictions: Dict, metrics: Dict, 
                         event_metrics: Optional[Dict],
                         output_dir: str):
    """
    Create comprehensive visualization plots
    
    Args:
        predictions: Dictionary with train/val/test predictions
        metrics: Performance metrics
        event_metrics: Event detection metrics (optional)
        output_dir: Directory to save plots
    """
    print("\n" + "="*80)
    print("📊 CREATING VISUALIZATIONS")
    print("="*80)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (20, 12)
    
    # ========================================================================
    # FIGURE 1: Performance Dashboard
    # ========================================================================
    fig1 = plt.figure(figsize=(20, 14))
    gs1 = plt.GridSpec(4, 2, figure=fig1, hspace=0.3, wspace=0.3)
    
    # Test predictions time series
    ax1 = fig1.add_subplot(gs1[0, :])
    test_actual = predictions['test']['actual']
    test_pred = predictions['test']['predicted']
    
    ax1.plot(test_actual.index, test_actual, label='Actual', color='blue', alpha=0.7, linewidth=1)
    ax1.plot(test_pred.index, test_pred, label='Predicted', color='red', alpha=0.7, linewidth=1)
    ax1.set_title(f'Test Set: Actual vs Predicted\nMAE: {metrics["test"]["MAE"]:.2f}, RMSE: {metrics["test"]["RMSE"]:.2f}, R²: {metrics["test"]["R2"]:.4f}',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Power (kW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation predictions
    ax2 = fig1.add_subplot(gs1[1, :])
    val_actual = predictions['val']['actual']
    val_pred = predictions['val']['predicted']
    
    ax2.plot(val_actual.index, val_actual, label='Actual', color='blue', alpha=0.7, linewidth=1)
    ax2.plot(val_pred.index, val_pred, label='Predicted', color='red', alpha=0.7, linewidth=1)
    ax2.set_title(f'Validation Set: Actual vs Predicted\nMAE: {metrics["val"]["MAE"]:.2f}, RMSE: {metrics["val"]["RMSE"]:.2f}, R²: {metrics["val"]["R2"]:.4f}',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Power (kW)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Scatter plot
    ax3 = fig1.add_subplot(gs1[2, 0])
    ax3.scatter(test_actual, test_pred, alpha=0.3, s=10, color='blue', label='Test')
    min_val = min(test_actual.min(), test_pred.min())
    max_val = max(test_actual.max(), test_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect')
    ax3.set_xlabel('Actual Power (kW)')
    ax3.set_ylabel('Predicted Power (kW)')
    ax3.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Residuals distribution
    ax4 = fig1.add_subplot(gs1[2, 1])
    residuals = test_actual - test_pred
    ax4.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Residual (kW)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Metrics comparison
    ax5 = fig1.add_subplot(gs1[3, :])
    metrics_df = pd.DataFrame({
        'Train': [metrics['train']['MAE'], metrics['train']['RMSE'], metrics['train']['R2']],
        'Val': [metrics['val']['MAE'], metrics['val']['RMSE'], metrics['val']['R2']],
        'Test': [metrics['test']['MAE'], metrics['test']['RMSE'], metrics['test']['R2']]
    }, index=['MAE', 'RMSE', 'R²'])
    
    x = np.arange(len(metrics_df.index))
    width = 0.25
    ax5.bar(x - width, metrics_df['Train'], width, label='Train', color='blue', alpha=0.7)
    ax5.bar(x, metrics_df['Val'], width, label='Val', color='green', alpha=0.7)
    ax5.bar(x + width, metrics_df['Test'], width, label='Test', color='red', alpha=0.7)
    ax5.set_ylabel('Value')
    ax5.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics_df.index)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('SARIMAX Model Performance Dashboard', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(os.path.join(output_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: performance_dashboard.png")
    
    # ========================================================================
    # FIGURE 2: Event Detection (if available)
    # ========================================================================
    if event_metrics is not None:
        fig2 = plt.figure(figsize=(20, 10))
        gs2 = plt.GridSpec(2, 3, figure=fig2, hspace=0.3, wspace=0.3)
        
        # Event detection metrics bar chart
        ax6 = fig2.add_subplot(gs2[0, :])
        
        categories = ['Overall', 'Significant', 'Stationary']
        precision_vals = [
            event_metrics['overall']['precision'],
            event_metrics['by_type'].get('significant', {}).get('precision', 0),
            event_metrics['by_type'].get('stationary', {}).get('precision', 0)
        ]
        recall_vals = [
            event_metrics['overall']['recall'],
            event_metrics['by_type'].get('significant', {}).get('recall', 0),
            event_metrics['by_type'].get('stationary', {}).get('recall', 0)
        ]
        f1_vals = [
            event_metrics['overall']['f1'],
            event_metrics['by_type'].get('significant', {}).get('f1', 0),
            event_metrics['by_type'].get('stationary', {}).get('f1', 0)
        ]
        
        x = np.arange(len(categories))
        width = 0.25
        
        ax6.bar(x - width, precision_vals, width, label='Precision', color='#2ecc71', alpha=0.8)
        ax6.bar(x, recall_vals, width, label='Recall', color='#3498db', alpha=0.8)
        ax6.bar(x + width, f1_vals, width, label='F1-Score', color='#e74c3c', alpha=0.8)
        
        ax6.set_ylabel('Score', fontsize=12)
        ax6.set_title('Event Detection Performance by Type', fontsize=14, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(categories)
        ax6.legend(fontsize=11)
        ax6.set_ylim([0, 1.0])
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (p, r, f) in enumerate(zip(precision_vals, recall_vals, f1_vals)):
            ax6.text(i - width, p + 0.02, f'{p:.3f}', ha='center', va='bottom', fontsize=9)
            ax6.text(i, r + 0.02, f'{r:.3f}', ha='center', va='bottom', fontsize=9)
            ax6.text(i + width, f + 0.02, f'{f:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Confusion matrix
        ax7 = fig2.add_subplot(gs2[1, 0])
        tp = event_metrics['overall']['true_positives']
        fp = event_metrics['overall']['false_positives']
        fn = event_metrics['overall']['false_negatives']
        cm_data = np.array([[tp, fn], [fp, 0]])
        
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax7,
                    xticklabels=['Pred Event', 'Pred No Event'],
                    yticklabels=['Actual Event', 'Actual No Event'])
        ax7.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        
        # Precision-Recall scatter
        ax8 = fig2.add_subplot(gs2[1, 1])
        for metric_name, color, marker in zip(['Overall', 'Significant', 'Stationary'],
                                               ['purple', 'orange', 'green'],
                                               ['o', 's', '^']):
            if metric_name == 'Overall':
                p = event_metrics['overall']['precision']
                r = event_metrics['overall']['recall']
                f = event_metrics['overall']['f1']
            else:
                p = event_metrics['by_type'].get(metric_name.lower(), {}).get('precision', 0)
                r = event_metrics['by_type'].get(metric_name.lower(), {}).get('recall', 0)
                f = event_metrics['by_type'].get(metric_name.lower(), {}).get('f1', 0)
            
            ax8.scatter(r, p, s=200, alpha=0.6, color=color, marker=marker,
                       label=f'{metric_name} (F1={f:.3f})', edgecolors='black', linewidth=2)
        
        ax8.set_xlabel('Recall', fontsize=11)
        ax8.set_ylabel('Precision', fontsize=11)
        ax8.set_title('Precision-Recall Trade-off', fontsize=12, fontweight='bold')
        ax8.set_xlim([0, 1.05])
        ax8.set_ylim([0, 1.05])
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        ax8.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        
        # Event counts comparison
        ax9 = fig2.add_subplot(gs2[1, 2])
        # Placeholder - would need actual/predicted event counts
        ax9.text(0.5, 0.5, 'Event counts\ncomparison', 
                ha='center', va='center', fontsize=12)
        ax9.set_title('Event Counts', fontsize=12, fontweight='bold')
        ax9.axis('off')
        
        plt.suptitle('Event Detection Performance Analysis', fontsize=18, fontweight='bold', y=0.995)
        plt.savefig(os.path.join(output_dir, 'event_detection_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: event_detection_analysis.png")


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(predictions: Dict, metrics: Dict, event_metrics: Optional[Dict],
                features: List[str], config: SARIMAXConfig, 
                sarimax_results, output_dir: str):
    """
    Save predictions and results to CSV and pickle files
    
    Args:
        predictions: Train/val/test predictions
        metrics: Performance metrics
        event_metrics: Event detection metrics (optional)
        features: Selected features list
        config: Configuration object
        sarimax_results: Fitted SARIMAX model results
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # ========================================================================
    # 1. Save predictions as CSV
    # ========================================================================
    predictions_df = pd.DataFrame({
        'actual': predictions['test']['actual'],
        'predicted': predictions['test']['predicted']
    })
    
    csv_path = os.path.join(output_dir, f'predictions_{timestamp}.csv')
    predictions_df.to_csv(csv_path, index=True)
    print(f"   ✓ CSV saved: predictions_{timestamp}.csv")
    
    # ========================================================================
    # 2. Save complete results as pickle
    # ========================================================================
    results_dict = {
        'predictions': predictions,
        'metrics': metrics,
        'event_metrics': event_metrics,
        'features': features,
        'config': {
            'ARIMA_ORDER': config.ARIMA_ORDER,
            'SEASONAL_ORDER': config.SEASONAL_ORDER,
            'MAX_FEATURES': config.MAX_FEATURES,
            'TRAIN_RATIO': config.TRAIN_RATIO,
            'VAL_RATIO': config.VAL_RATIO,
            'TEST_RATIO': config.TEST_RATIO
        },
        'model_summary': {
            'AIC': sarimax_results.aic,
            'BIC': sarimax_results.bic,
            'log_likelihood': sarimax_results.llf,
            'converged': sarimax_results.mle_retvals.get('converged', False)
        },
        'timestamp': timestamp
    }
    
    pkl_path = os.path.join(output_dir, f'results_{timestamp}.pkl')
    import pickle
    with open(pkl_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"   ✓ Pickle saved: results_{timestamp}.pkl")
    
    # ========================================================================
    # 3. Save metrics summary as JSON
    # ========================================================================
    import json
    
    metrics_summary = {
        'performance': {
            'train': {k: float(v) for k, v in metrics['train'].items()},
            'val': {k: float(v) for k, v in metrics['val'].items()},
            'test': {k: float(v) for k, v in metrics['test'].items()}
        },
        'model': {
            'AIC': float(sarimax_results.aic),
            'BIC': float(sarimax_results.bic),
            'log_likelihood': float(sarimax_results.llf),
            'converged': bool(sarimax_results.mle_retvals.get('converged', False))
        },
        'features': {
            'count': len(features),
            'list': features
        }
    }
    
    if event_metrics is not None:
        metrics_summary['event_detection'] = {
            'overall': {k: float(v) if isinstance(v, (int, float)) else v 
                       for k, v in event_metrics['overall'].items()},
            'by_type': {
                event_type: {k: float(v) if isinstance(v, (int, float)) else v 
                            for k, v in type_metrics.items()}
                for event_type, type_metrics in event_metrics['by_type'].items()
            }
        }
    
    json_path = os.path.join(output_dir, f'metrics_summary_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"   ✓ JSON saved: metrics_summary_{timestamp}.json")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main(data_path: str, output_dir: Optional[str] = None):
    """
    Main SARIMAX-RBA workflow
    
    Args:
        data_path: Path to input data
        output_dir: Output directory (optional)
    
    Returns:
        Dictionary with all results
    """
    # Initialize config
    config = SARIMAXConfig()
    
    if output_dir is not None:
        config.OUTPUT_DIR = output_dir
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("🚀 SARIMAX-RBA WORKFLOW")
    print("="*80)
    print(f"\nData: {data_path}")
    print(f"Output: {config.OUTPUT_DIR}")
    print(f"Model: ARIMA{config.ARIMA_ORDER} + Seasonal{config.SEASONAL_ORDER}")
    
    # Load data
    data = load_and_preprocess_data(data_path, config)
    nominal = data[config.TARGET_COLUMN].max()
    
    # Extract RBA events
    sig_events, stat_events = extract_rba_events(data, nominal)
    events_df = convert_events_to_dataframe(sig_events, stat_events)
    
    # Split data
    splits = split_data_and_events(data, events_df, config)
    
    # Engineer features
    train_features = create_universal_features(
        splits['train_raw'], splits['train_events'],
        splits['full_data'].loc[splits['train_raw'].index],
        "Training", config
    )
    
    val_features = create_universal_features(
        splits['val_raw'], splits['val_events'],
        splits['full_data'].loc[splits['val_raw'].index],
        "Validation", config
    )
    
    test_features = create_universal_features(
        splits['test_raw'], splits['test_events'],
        splits['full_data'].loc[splits['test_raw'].index],
        "Test", config
    )
    
    # Detect seasonal period
    seasonal_m = infer_seasonal_period(train_features.index)
    print(f"\n📊 Detected seasonal period: {seasonal_m}")
    
    # Select features
    exog_cols = select_exog_features(train_features, config.TARGET_COLUMN, 
                                     config.MAX_FEATURES)
    
    # Build train/val/test datasets
    y_train, X_train, scaler = build_endog_exog(
        train_features, config.TARGET_COLUMN, exog_cols, 
        scaler=None, fit_scaler=True
    )
    
    y_val, X_val, _ = build_endog_exog(
        val_features, config.TARGET_COLUMN, exog_cols,
        scaler=scaler, fit_scaler=False
    )
    
    y_test, X_test, _ = build_endog_exog(
        test_features, config.TARGET_COLUMN, exog_cols,
        scaler=scaler, fit_scaler=False
    )
    
    # Train model
    results = train_sarimax_model(y_train, X_train, config, seasonal_m)
    
    # Make predictions
    pred_results = make_predictions(
        results, y_train, y_val, y_test,
        X_train, X_val, X_test
    )
    
    # ========================================================================
    # POST-HOC EVENT EXTRACTION AND EVALUATION
    # ========================================================================
    event_metrics = None
    
    if CORE_AVAILABLE and not events_df.empty:
        print("\n" + "="*80)
        print("🎯 POST-HOC EVENT EXTRACTION FROM PREDICTIONS")
        print("="*80)
        
        # Extract events from predicted test time series
        test_predicted_series = pred_results['predictions']['test']['predicted']
        predicted_events = extract_events_from_predictions(
            test_predicted_series, 
            nominal,
            turbine_id='Turbine_1'
        )
        
        # Get actual test events
        actual_test_events = splits['test_events']
        
        # Evaluate event detection performance
        if not predicted_events.empty and not actual_test_events.empty:
            event_metrics = evaluate_event_detection(
                actual_test_events,
                predicted_events,
                tolerance=5
            )
    
    # ========================================================================
    # CREATE VISUALIZATIONS
    # ========================================================================
    if config.SAVE_PLOTS:
        create_visualizations(
            pred_results['predictions'],
            pred_results['metrics'],
            event_metrics,
            config.OUTPUT_DIR
        )
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    if config.SAVE_RESULTS:
        save_results(
            pred_results['predictions'],
            pred_results['metrics'],
            event_metrics,
            exog_cols,
            config,
            results,
            config.OUTPUT_DIR
        )
    
    print("\n✅ Workflow complete!")
    
    return {
        'config': config,
        'results': results,
        'predictions': pred_results['predictions'],
        'metrics': pred_results['metrics'],
        'event_metrics': event_metrics,
        'features': exog_cols,
        'seasonal_m': seasonal_m,
        'splits': splits,
        'events': events_df,
        'output_dir': config.OUTPUT_DIR
    }


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SARIMAX-RBA Wind Power Forecasting (Workflow1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training with fixed ARIMA(1,0,1) and no seasonality
  python workflows/workflow1.py --data input_data/Baltic_Eagle.xlsx
  
  # Custom output directory
  python workflows/workflow1.py --data input_data/Baltic_Eagle.xlsx --output results/

Model Configuration:
  - ARIMA order: (1, 0, 1)
  - Seasonal order: (0, 0, 0, 0) - No seasonality
  - To change these, edit SARIMAXConfig class in the code
        """
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to input data (Excel file)')
    
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: sarimax_predictions)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        sys.exit(1)
    
    # Run workflow
    try:
        result = main(
            data_path=args.data,
            output_dir=args.output
        )
        
        print("\n" + "="*80)
        print("✅ SUCCESS")
        print("="*80)
        print(f"\n📊 Final Test Performance:")
        print(f"   MAE:  {result['metrics']['test']['MAE']:.4f}")
        print(f"   RMSE: {result['metrics']['test']['RMSE']:.4f}")
        print(f"   R²:   {result['metrics']['test']['R2']:.4f}")
        
        if result.get('event_metrics') is not None:
            print(f"\n🎯 Event Detection Performance:")
            print(f"   Precision: {result['event_metrics']['overall']['precision']:.4f}")
            print(f"   Recall:    {result['event_metrics']['overall']['recall']:.4f}")
            print(f"   F1-Score:  {result['event_metrics']['overall']['f1']:.4f}")
        
        print(f"\n💾 Results saved to: {result['output_dir']}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
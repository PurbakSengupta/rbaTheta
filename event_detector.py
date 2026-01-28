"""
Enhanced Event Detector - Multi-Method Comparison with Flexible Data Handling
Runs Enhanced RBA-theta, Classic RBA-theta, CUSUM, SWRT, and Adaptive methods together
Key improvements: 
- Flexible datetime column detection
- CLI compatibility with main orchestrator
- Comprehensive comparison with turbine-by-turbine results
"""

import time
import os
import sys
import argparse
import multiprocessing
import pandas as pd
import numpy as np
from core import save_xls
import core.model as model
from core.model import tune_mixed_strategy
import logging

# Import all comparison methods
try:
    import core.classic_model as classic_model
    CLASSIC_RBA_AVAILABLE = True
except ImportError:
    print("⚠️  classic_model.py not found - Classic RBA-theta will be skipped")
    CLASSIC_RBA_AVAILABLE = False

try:
    from core.cusum_method import run_cusum_analysis
    CUSUM_AVAILABLE = True
except ImportError:
    print("⚠️  cusum_method.py not found - CUSUM will be skipped")
    CUSUM_AVAILABLE = False

try:
    from core.swrt_method import run_swrt_analysis
    SWRT_AVAILABLE = True
except ImportError:
    print("⚠️  swrt_method.py not found - SWRT will be skipped")
    SWRT_AVAILABLE = False

try:
    from core.adaptive_baselines import run_adaptive_cusum_analysis, run_adaptive_swrt_analysis
    ADAPTIVE_AVAILABLE = True
except ImportError:
    print("⚠️  adaptive_baselines.py not found - Adaptive methods will be skipped")
    ADAPTIVE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# FLEXIBLE DATETIME HANDLING
# ============================================================================

def detect_and_parse_datetime(df, preferred_col=None):
    """
    Flexibly detect and parse datetime column from various naming conventions
    
    Args:
        df: Input DataFrame
        preferred_col: Preferred column name (if known)
    
    Returns:
        DataFrame with datetime index
    """
    # Common datetime column names (in order of preference)
    datetime_variants = [
        'DateTime', 'datetime', 'Datetime', 'DATETIME',
        'Time', 'time', 'TIME',
        'Date', 'date', 'DATE',
        'timestamp', 'Timestamp', 'TIMESTAMP',
        'Date Time', 'date time', 'DATE TIME',
        'date_time', 'Date_Time', 'DATE_TIME'
    ]
    
    # If preferred column is specified, try it first
    if preferred_col is not None:
        if preferred_col in df.columns:
            datetime_variants.insert(0, preferred_col)
    
    # Find the datetime column
    datetime_col = None
    for col in datetime_variants:
        if col in df.columns:
            datetime_col = col
            logger.info(f"✓ Found datetime column: '{col}'")
            break
    
    # If not found by name, search by dtype
    if datetime_col is None:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_col = col
                logger.info(f"✓ Found datetime column by dtype: '{col}'")
                break
    
    # If still not found, look for parseable date strings
    if datetime_col is None:
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Try to parse a sample
                    sample = df[col].dropna().iloc[0]
                    pd.to_datetime(sample)
                    datetime_col = col
                    logger.info(f"✓ Found parseable datetime column: '{col}'")
                    break
                except:
                    continue
    
    if datetime_col is None:
        raise ValueError(f"No datetime column found. Available columns: {list(df.columns)}")
    
    # Parse datetime with multiple strategies
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        logger.info(f"Parsing datetime column '{datetime_col}'...")
        
        # Strategy 1: Try dayfirst=True (European format)
        try:
            df[datetime_col] = pd.to_datetime(df[datetime_col], dayfirst=True, errors='raise')
            logger.info("✓ Parsed with dayfirst=True")
        except:
            # Strategy 2: Try dayfirst=False (US format)
            try:
                df[datetime_col] = pd.to_datetime(df[datetime_col], dayfirst=False, errors='raise')
                logger.info("✓ Parsed with dayfirst=False")
            except:
                # Strategy 3: Try ISO format
                try:
                    df[datetime_col] = pd.to_datetime(df[datetime_col], format='ISO8601', errors='raise')
                    logger.info("✓ Parsed with ISO8601 format")
                except:
                    # Strategy 4: Let pandas infer format with coerce
                    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
                    logger.info("✓ Parsed with automatic format detection")
                    
                    # Check for failed parses
                    if df[datetime_col].isna().any():
                        failed_count = df[datetime_col].isna().sum()
                        logger.warning(f"⚠️  {failed_count} timestamps failed to parse")
    
    # Set as index
    df = df.set_index(datetime_col)
    df = df.sort_index()
    
    logger.info(f"✓ Datetime index set: {df.index[0]} to {df.index[-1]}")
    
    return df


# ============================================================================
# TURBINE IDENTIFICATION
# ============================================================================

def add_turbine_id_if_missing(events_df, method_name="Unknown Method"):
    """
    Add turbine_id to events DataFrame if not present
    """
    if events_df.empty:
        return events_df
    
    # Check if turbine_id already exists
    turbine_cols = ['turbine_id', 'Turbine_ID', 'turbine', 'Turbine', 'turbine_number', 'turbine_column']
    has_turbine_col = any(col in events_df.columns for col in turbine_cols)
    
    if not has_turbine_col:
        logger.info(f"Adding turbine_id to {method_name} events")
        events_with_turbine = events_df.copy()
        
        # Method 1: Check for index-based patterns
        if 'start_index' in events_df.columns:
            events_with_turbine['turbine_id'] = ((events_df['start_index'] % 8) + 1).astype(int)
            logger.info(f"Used start_index modulo 8 for turbine assignment")
        elif 'end_index' in events_df.columns:
            events_with_turbine['turbine_id'] = ((events_df['end_index'] % 8) + 1).astype(int)
            logger.info(f"Used end_index modulo 8 for turbine assignment")
        elif 'index' in events_df.columns:
            events_with_turbine['turbine_id'] = ((events_df['index'] % 8) + 1).astype(int)
            logger.info(f"Used index modulo 8 for turbine assignment")
        else:
            # Method 2: Look for numeric columns with values 1-8
            numeric_cols = events_df.select_dtypes(include=['int64', 'float64']).columns
            turbine_assigned = False
            
            for col in numeric_cols:
                try:
                    unique_vals = set(events_df[col].dropna().astype(int))
                    if unique_vals.issubset(set(range(1, 9))) and len(unique_vals) > 1:
                        events_with_turbine['turbine_id'] = events_df[col].astype(int)
                        logger.info(f"Using column '{col}' as turbine identifier")
                        turbine_assigned = True
                        break
                except:
                    continue
            
            if not turbine_assigned:
                # Method 3: Distribute events evenly
                num_events = len(events_df)
                turbine_assignment = [(i % 8) + 1 for i in range(num_events)]
                events_with_turbine['turbine_id'] = turbine_assignment
                logger.info(f"Distributing {num_events} events evenly across 8 turbines")
        
        return events_with_turbine
    else:
        # Turbine column exists - ensure it's integer
        existing_turbine_col = None
        for col in turbine_cols:
            if col in events_df.columns:
                existing_turbine_col = col
                break
        
        if existing_turbine_col:
            events_with_turbine = events_df.copy()
            try:
                events_with_turbine['turbine_id'] = events_with_turbine[existing_turbine_col].astype(int)
                if existing_turbine_col != 'turbine_id':
                    events_with_turbine = events_with_turbine.drop(columns=[existing_turbine_col])
                logger.info(f"Converted existing turbine column '{existing_turbine_col}' to integer turbine_id")
                return events_with_turbine
            except:
                # Extract numbers from strings
                def extract_turbine_number(val):
                    if pd.isna(val):
                        return 1
                    val_str = str(val)
                    import re
                    numbers = re.findall(r'\d+', val_str)
                    if numbers:
                        num = int(numbers[0])
                        return num if 1 <= num <= 8 else ((num - 1) % 8) + 1
                    return 1
                
                events_with_turbine['turbine_id'] = events_with_turbine[existing_turbine_col].apply(extract_turbine_number)
                if existing_turbine_col != 'turbine_id':
                    events_with_turbine = events_with_turbine.drop(columns=[existing_turbine_col])
                logger.info(f"Extracted numbers from turbine column '{existing_turbine_col}'")
                return events_with_turbine
    
    return events_df


def save_events_by_turbine(events_df, filepath, sheet_prefix="Turbine"):
    """
    Save events separated by turbine in different sheets
    
    Args:
        events_df: DataFrame with events
        filepath: Path to save the Excel file
        sheet_prefix: Prefix for sheet names
    """
    if events_df.empty:
        # Create empty file with 8 empty sheets for consistency
        empty_sheets = {f"{sheet_prefix}_{i}": pd.DataFrame() for i in range(1, 9)}
        save_xls(empty_sheets, filepath)
        return
    
    # Ensure turbine_id exists
    if 'turbine_id' not in events_df.columns:
        events_df = add_turbine_id_if_missing(events_df, "Unknown")
    
    # Group events by turbine
    turbine_sheets = {}
    
    # Create sheets for all 8 turbines
    for i in range(1, 9):
        turbine_events = events_df[events_df['turbine_id'] == i]
        if not turbine_events.empty:
            # Remove turbine_id column from individual sheets
            turbine_events_clean = turbine_events.drop(columns=['turbine_id'])
            turbine_sheets[f"{sheet_prefix}_{i}"] = turbine_events_clean
        else:
            turbine_sheets[f"{sheet_prefix}_{i}"] = pd.DataFrame()
    
    # Save all sheets to Excel
    save_xls(turbine_sheets, filepath)
    
    # Log summary
    non_empty_turbines = sum(1 for df in turbine_sheets.values() if not df.empty)
    total_events = sum(len(df) for df in turbine_sheets.values())
    logger.info(f"Saved {total_events} events across {non_empty_turbines} turbines to {filepath}")


# ============================================================================
# INTERACTIVE COLUMN SELECTION
# ============================================================================

def detect_numerical_columns(df):
    """
    Detect numerical columns suitable for event detection
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        List of numerical column names
    """
    # Get numerical columns (excluding index)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude columns that look like IDs or indices
    excluded_patterns = ['unnamed', 'index', 'id', 'row', 'count', 'number']
    filtered_cols = [
        col for col in numerical_cols 
        if not any(pattern in col.lower() for pattern in excluded_patterns)
    ]
    
    return filtered_cols


def prompt_column_selection(df, interactive=True):
    """
    Interactively prompt user to select columns for analysis
    
    Args:
        df: DataFrame with datetime index
        interactive: Whether to use interactive prompts (False for non-interactive mode)
    
    Returns:
        Tuple of (analysis_type, selected_columns)
        - analysis_type: 'univariate' or 'multivariate'
        - selected_columns: List of column names
    """
    numerical_cols = detect_numerical_columns(df)
    
    if len(numerical_cols) == 0:
        raise ValueError("No numerical columns found in dataset suitable for analysis")
    
    print("\n" + "="*80)
    print("📊 COLUMN SELECTION FOR EVENT DETECTION")
    print("="*80)
    
    print(f"\n✓ Found {len(numerical_cols)} numerical column(s):")
    for i, col in enumerate(numerical_cols, 1):
        # Show sample statistics
        col_data = df[col].dropna()
        print(f"   {i}. {col:<30} (Mean: {col_data.mean():.2f}, Std: {col_data.std():.2f}, Range: [{col_data.min():.2f}, {col_data.max():.2f}])")
    
    if not interactive:
        # Non-interactive mode: use first column for univariate
        print(f"\n⚡ Non-interactive mode: Using '{numerical_cols[0]}' for univariate analysis")
        return 'univariate', [numerical_cols[0]]
    
    # Interactive mode
    print("\n" + "-"*80)
    print("Analysis Type Selection:")
    print("   1. Univariate  - Analyze a single column (e.g., Power, Windspeed)")
    print("   2. Multivariate - Analyze multiple columns simultaneously")
    print("-"*80)
    
    # Get analysis type
    while True:
        try:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            
            if choice == '1':
                analysis_type = 'univariate'
                print("\n✓ Selected: Univariate Analysis")
                break
            elif choice == '2':
                analysis_type = 'multivariate'
                print("\n✓ Selected: Multivariate Analysis")
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Operation cancelled by user")
            sys.exit(0)
    
    # Get column selection
    if analysis_type == 'univariate':
        print("\n" + "-"*80)
        print("Enter the column name to analyze:")
        print("(Type the exact column name from the list above)")
        print("-"*80)
        
        while True:
            try:
                column_input = input("\nColumn name: ").strip()
                
                if not column_input:
                    print("❌ Column name cannot be empty. Please try again.")
                    continue
                
                # Check if user entered multiple columns (comma-separated)
                if ',' in column_input:
                    columns = [col.strip() for col in column_input.split(',')]
                    print(f"\n⚠️  You entered {len(columns)} columns: {columns}")
                    print("💡 For multiple columns, please use option 2 (Multivariate Analysis)")
                    
                    redirect = input("\nWould you like to switch to multivariate analysis? (y/n): ").strip().lower()
                    if redirect == 'y':
                        analysis_type = 'multivariate'
                        print("\n✓ Switched to Multivariate Analysis")
                        # Continue to multivariate selection below
                        break
                    else:
                        print("\n📌 Please enter a single column name for univariate analysis:")
                        continue
                
                # Validate single column
                if column_input not in numerical_cols:
                    print(f"❌ Column '{column_input}' not found.")
                    print(f"   Available columns: {', '.join(numerical_cols)}")
                    continue
                
                selected_columns = [column_input]
                print(f"\n✓ Selected column: '{column_input}'")
                return analysis_type, selected_columns
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Operation cancelled by user")
                sys.exit(0)
    
    # Multivariate selection
    if analysis_type == 'multivariate':
        print("\n" + "-"*80)
        print("Enter column names to analyze (comma-separated):")
        print("Examples:")
        print("  - Power, Windspeed")
        print("  - Power, Windspeed, Wind_Direction")
        print("  - all  (to analyze all numerical columns)")
        print("-"*80)
        
        while True:
            try:
                column_input = input("\nColumn names: ").strip()
                
                if not column_input:
                    print("❌ Input cannot be empty. Please try again.")
                    continue
                
                # Check for 'all' keyword
                if column_input.lower() == 'all':
                    selected_columns = numerical_cols
                    print(f"\n✓ Selected ALL {len(selected_columns)} columns:")
                    for col in selected_columns:
                        print(f"   - {col}")
                    return analysis_type, selected_columns
                
                # Parse comma-separated columns
                columns = [col.strip() for col in column_input.split(',')]
                
                # Validate each column
                invalid_cols = [col for col in columns if col not in numerical_cols]
                if invalid_cols:
                    print(f"\n❌ Column(s) not found: {', '.join(invalid_cols)}")
                    print(f"   Available columns: {', '.join(numerical_cols)}")
                    continue
                
                selected_columns = columns
                print(f"\n✓ Selected {len(selected_columns)} column(s):")
                for col in selected_columns:
                    print(f"   - {col}")
                
                return analysis_type, selected_columns
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Operation cancelled by user")
                sys.exit(0)


def prepare_data_for_analysis(df, selected_columns, analysis_type):
    """
    Prepare data based on selected columns and analysis type
    
    Args:
        df: DataFrame with datetime index
        selected_columns: List of column names
        analysis_type: 'univariate' or 'multivariate'
    
    Returns:
        DataFrame ready for analysis
    """
    if analysis_type == 'univariate':
        # Rename single column to 'Turbine_1' for compatibility with RBA
        analysis_df = df[[selected_columns[0]]].copy()
        analysis_df = analysis_df.rename(columns={selected_columns[0]: 'Turbine_1'})
        logger.info(f"Prepared univariate data: '{selected_columns[0]}' → 'Turbine_1'")
    
    else:  # multivariate
        # Rename multiple columns to 'Turbine_1', 'Turbine_2', etc.
        analysis_df = df[selected_columns].copy()
        
        rename_mapping = {
            col: f'Turbine_{i+1}' 
            for i, col in enumerate(selected_columns)
        }
        
        analysis_df = analysis_df.rename(columns=rename_mapping)
        logger.info(f"Prepared multivariate data: {len(selected_columns)} columns → Turbine_1 to Turbine_{len(selected_columns)}")
        
        # Log the mapping
        for original, renamed in rename_mapping.items():
            logger.info(f"  {original} → {renamed}")
    
    return analysis_df


# ============================================================================
# COMPREHENSIVE ANALYSIS
# ============================================================================

def comprehensive_analysis(data_path, use_optimization=True, output_dir='simulations/all_tests_together',
                          interactive=True, selected_columns=None, analysis_type=None):
    """
    Run all available methods on the same dataset for comprehensive comparison
    
    Args:
        data_path: Path to input data file
        use_optimization: Whether to optimize parameters
        output_dir: Directory to save results
        interactive: Whether to prompt for column selection
        selected_columns: Pre-selected columns (for non-interactive mode)
        analysis_type: Pre-selected analysis type (for non-interactive mode)
    
    Returns:
        Dictionary with results summary
    """
    print("\n" + "="*80)
    print("🌪️  COMPREHENSIVE WIND TURBINE EVENT DETECTION ANALYSIS")
    print("="*80)
    
    # Start total timing
    total_start_time = time.time()
    results_summary = {}
    method_times = {}
    
    try:
        # ====================================================================
        # LOAD AND PREPARE DATA
        # ====================================================================
        print("\n📊 Loading and preparing data...")
        data_load_start = time.time()
        
        # Load data
        wind_data = pd.read_excel(data_path)
        logger.info(f"Raw data loaded: {wind_data.shape}")
        logger.info(f"Columns: {list(wind_data.columns)}")
        
        # Flexibly detect and parse datetime
        wind_data = detect_and_parse_datetime(wind_data)
        
        # ====================================================================
        # COLUMN SELECTION
        # ====================================================================
        # If columns not pre-selected, prompt user
        if selected_columns is None or analysis_type is None:
            analysis_type, selected_columns = prompt_column_selection(wind_data, interactive=interactive)
        
        # Prepare data based on selection
        wind_data_analysis = prepare_data_for_analysis(wind_data, selected_columns, analysis_type)
        
        data_load_time = time.time() - data_load_start
        
        # Calculate nominal value
        nominal = wind_data_analysis.select_dtypes(include='number').max().max()
        logger.info(f"Data prepared: {len(wind_data_analysis)} records, nominal value: {nominal:.3f}")
        logger.info(f"⏱️  Data loading time: {data_load_time:.2f} seconds")
        
        print(f"\n✓ Analysis ready:")
        print(f"   Type: {analysis_type.upper()}")
        print(f"   Columns: {selected_columns}")
        print(f"   Time range: {wind_data_analysis.index[0]} to {wind_data_analysis.index[-1]}")
        print(f"   Data points: {len(wind_data_analysis):,}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # ====================================================================
        # METHOD 1: ENHANCED RBA-THETA
        # ====================================================================
        print("\n🚀 1. Running Enhanced RBA-theta (Current Implementation)...")
        enhanced_start = time.time()
        
        try:
            # Get optimal configuration
            if use_optimization:
                logger.info("   Optimizing parameters for Enhanced RBA-theta...")
                best_config = tune_mixed_strategy(wind_data_analysis, nominal)
                logger.info("   Parameter optimization completed")
            else:
                logger.info("   Using default parameters")
                best_config = None

            # Run Enhanced RBA_theta analysis
            enhanced_results = model.RBA_theta(wind_data_analysis, nominal, best_config)
            enhanced_sig_trad, enhanced_stat_trad, enhanced_sig_mcmc, enhanced_stat_mcmc, enhanced_tao = enhanced_results

            # Add turbine identification
            enhanced_sig_trad = add_turbine_id_if_missing(enhanced_sig_trad, "Enhanced RBA Traditional Significant")
            enhanced_stat_trad = add_turbine_id_if_missing(enhanced_stat_trad, "Enhanced RBA Traditional Stationary")
            enhanced_sig_mcmc = add_turbine_id_if_missing(enhanced_sig_mcmc, "Enhanced RBA MCMC Significant")
            enhanced_stat_mcmc = add_turbine_id_if_missing(enhanced_stat_mcmc, "Enhanced RBA MCMC Stationary")

            # Calculate quality metrics
            enhanced_trad_metrics = model.calculate_event_quality_metrics(enhanced_sig_trad, enhanced_stat_trad)
            enhanced_mcmc_metrics = model.calculate_event_quality_metrics(enhanced_sig_mcmc, enhanced_stat_mcmc)
            
            enhanced_time = time.time() - enhanced_start
            method_times['Enhanced RBA-theta'] = enhanced_time
            
            # Save Enhanced RBA-theta results BY TURBINE
            save_events_by_turbine(enhanced_sig_trad, 
                                 os.path.join(output_dir, 'enhanced_rba_traditional_significant.xlsx'),
                                 'Turbine')
            save_events_by_turbine(enhanced_stat_trad, 
                                 os.path.join(output_dir, 'enhanced_rba_traditional_stationary.xlsx'),
                                 'Turbine')
            save_events_by_turbine(enhanced_sig_mcmc, 
                                 os.path.join(output_dir, 'enhanced_rba_mcmc_significant.xlsx'),
                                 'Turbine')
            save_events_by_turbine(enhanced_stat_mcmc, 
                                 os.path.join(output_dir, 'enhanced_rba_mcmc_stationary.xlsx'),
                                 'Turbine')
            
            enhanced_total_events = len(enhanced_sig_trad) + len(enhanced_stat_trad) + len(enhanced_sig_mcmc) + len(enhanced_stat_mcmc)
            results_summary['Enhanced RBA-theta'] = {
                'total_events': enhanced_total_events,
                'traditional_events': len(enhanced_sig_trad) + len(enhanced_stat_trad),
                'mcmc_events': len(enhanced_sig_mcmc) + len(enhanced_stat_mcmc),
                'trad_quality': enhanced_trad_metrics['overall']['balance_score'],
                'mcmc_quality': enhanced_mcmc_metrics['overall']['balance_score'],
                'time': enhanced_time,
                'status': 'Success'
            }
            
            print(f"   ✅ Enhanced RBA-theta completed: {enhanced_total_events} events in {enhanced_time:.2f}s")
            
        except Exception as e:
            enhanced_time = time.time() - enhanced_start
            method_times['Enhanced RBA-theta'] = enhanced_time
            results_summary['Enhanced RBA-theta'] = {'status': f'Failed: {e}', 'time': enhanced_time}
            print(f"   ❌ Enhanced RBA-theta failed: {e}")

        # ====================================================================
        # METHOD 2: CLASSIC RBA-THETA
        # ====================================================================
        if CLASSIC_RBA_AVAILABLE:
            print("\n📚 2. Running Classic RBA-theta (Original Implementation)...")
            classic_start = time.time()
            
            try:
                # Run Classic RBA_theta analysis
                classic_results = classic_model.RBA_theta(wind_data_analysis, nominal)
                classic_sig_events_dict, classic_stat_events_dict, classic_tao = classic_results

                # Convert dictionary format to DataFrame
                def convert_dict_to_dataframe(event_dict):
                    if not event_dict or all(df.empty for df in event_dict.values() if hasattr(df, 'empty')):
                        return pd.DataFrame()
                    valid_events = []
                    for turbine_id, events_df in event_dict.items():
                        if hasattr(events_df, 'empty') and not events_df.empty:
                            events_copy = events_df.copy()
                            # Extract turbine number
                            if isinstance(turbine_id, str) and '_' in turbine_id:
                                turbine_num = int(turbine_id.split('_')[-1])
                            elif isinstance(turbine_id, str) and turbine_id.isdigit():
                                turbine_num = int(turbine_id)
                            elif isinstance(turbine_id, (int, float)):
                                turbine_num = int(turbine_id)
                            else:
                                import re
                                numbers = re.findall(r'\d+', str(turbine_id))
                                turbine_num = int(numbers[0]) if numbers else 1
                            events_copy['turbine_id'] = turbine_num
                            valid_events.append(events_copy)
                    return pd.concat(valid_events, ignore_index=True) if valid_events else pd.DataFrame()

                classic_sig_trad = convert_dict_to_dataframe(classic_sig_events_dict)
                classic_stat_trad = convert_dict_to_dataframe(classic_stat_events_dict)
                
                # Calculate quality metrics
                if hasattr(model, 'calculate_event_quality_metrics'):
                    classic_trad_metrics = model.calculate_event_quality_metrics(classic_sig_trad, classic_stat_trad)
                else:
                    classic_trad_metrics = classic_model.calculate_quality_metrics(classic_sig_trad, classic_stat_trad)
                
                classic_time = time.time() - classic_start
                method_times['Classic RBA-theta'] = classic_time
                
                # Save Classic RBA-theta results
                save_events_by_turbine(classic_sig_trad, 
                                     os.path.join(output_dir, 'classic_rba_significant_events.xlsx'),
                                     'Turbine')
                save_events_by_turbine(classic_stat_trad, 
                                     os.path.join(output_dir, 'classic_rba_stationary_events.xlsx'),
                                     'Turbine')
                
                classic_total_events = len(classic_sig_trad) + len(classic_stat_trad)
                results_summary['Classic RBA-theta'] = {
                    'total_events': classic_total_events,
                    'traditional_events': classic_total_events,
                    'mcmc_events': 0,
                    'trad_quality': classic_trad_metrics['overall']['balance_score'],
                    'mcmc_quality': 0.0,
                    'time': classic_time,
                    'status': 'Success'
                }
                
                print(f"   ✅ Classic RBA-theta completed: {classic_total_events} events in {classic_time:.2f}s")
                
            except Exception as e:
                classic_time = time.time() - classic_start
                method_times['Classic RBA-theta'] = classic_time
                results_summary['Classic RBA-theta'] = {'status': f'Failed: {e}', 'time': classic_time}
                print(f"   ❌ Classic RBA-theta failed: {e}")
        else:
            print("\n⏭️  2. Classic RBA-theta skipped (not available)")

        # ====================================================================
        # METHOD 3: CUSUM
        # ====================================================================
        if CUSUM_AVAILABLE:
            print("\n📈 3. Running CUSUM Method...")
            cusum_start = time.time()
            
            try:
                cusum_events = run_cusum_analysis(wind_data_analysis)
                cusum_events = add_turbine_id_if_missing(cusum_events, "CUSUM")
                
                cusum_time = time.time() - cusum_start
                method_times['CUSUM'] = cusum_time
                
                save_events_by_turbine(cusum_events, 
                                     os.path.join(output_dir, 'cusum_events.xlsx'),
                                     'Turbine')
                
                cusum_total_events = len(cusum_events)
                results_summary['CUSUM'] = {
                    'total_events': cusum_total_events,
                    'time': cusum_time,
                    'status': 'Success'
                }
                
                print(f"   ✅ CUSUM completed: {cusum_total_events} events in {cusum_time:.2f}s")
                
            except Exception as e:
                cusum_time = time.time() - cusum_start
                method_times['CUSUM'] = cusum_time
                results_summary['CUSUM'] = {'status': f'Failed: {e}', 'time': cusum_time}
                print(f"   ❌ CUSUM failed: {e}")
        else:
            print("\n⏭️  3. CUSUM skipped (not available)")

        # ====================================================================
        # METHOD 4: SWRT
        # ====================================================================
        if SWRT_AVAILABLE:
            print("\n🌪️  4. Running SWRT Method...")
            swrt_start = time.time()
            
            try:
                swrt_events = run_swrt_analysis(wind_data_analysis, nominal)
                swrt_events = add_turbine_id_if_missing(swrt_events, "SWRT")
                
                swrt_time = time.time() - swrt_start
                method_times['SWRT'] = swrt_time
                
                save_events_by_turbine(swrt_events, 
                                     os.path.join(output_dir, 'swrt_events.xlsx'),
                                     'Turbine')
                
                swrt_total_events = len(swrt_events)
                results_summary['SWRT'] = {
                    'total_events': swrt_total_events,
                    'time': swrt_time,
                    'status': 'Success'
                }
                
                print(f"   ✅ SWRT completed: {swrt_total_events} events in {swrt_time:.2f}s")
                
            except Exception as e:
                swrt_time = time.time() - swrt_start
                method_times['SWRT'] = swrt_time
                results_summary['SWRT'] = {'status': f'Failed: {e}', 'time': swrt_time}
                print(f"   ❌ SWRT failed: {e}")
        else:
            print("\n⏭️  4. SWRT skipped (not available)")

        # ====================================================================
        # METHOD 5-6: ADAPTIVE METHODS
        # ====================================================================
        if ADAPTIVE_AVAILABLE:
            print("\n🔧 5. Running Adaptive CUSUM...")
            adaptive_cusum_start = time.time()
            
            try:
                adaptive_cusum_events = run_adaptive_cusum_analysis(wind_data_analysis)
                adaptive_cusum_events = add_turbine_id_if_missing(adaptive_cusum_events, "Adaptive CUSUM")
                
                adaptive_cusum_time = time.time() - adaptive_cusum_start
                method_times['Adaptive CUSUM'] = adaptive_cusum_time
                
                save_events_by_turbine(adaptive_cusum_events, 
                                     os.path.join(output_dir, 'adaptive_cusum_events.xlsx'),
                                     'Turbine')
                
                adaptive_cusum_total_events = len(adaptive_cusum_events)
                results_summary['Adaptive CUSUM'] = {
                    'total_events': adaptive_cusum_total_events,
                    'time': adaptive_cusum_time,
                    'status': 'Success'
                }
                
                print(f"   ✅ Adaptive CUSUM completed: {adaptive_cusum_total_events} events in {adaptive_cusum_time:.2f}s")
                
            except Exception as e:
                adaptive_cusum_time = time.time() - adaptive_cusum_start
                method_times['Adaptive CUSUM'] = adaptive_cusum_time
                results_summary['Adaptive CUSUM'] = {'status': f'Failed: {e}', 'time': adaptive_cusum_time}
                print(f"   ❌ Adaptive CUSUM failed: {e}")

            print("\n🌪️  6. Running Adaptive SWRT...")
            adaptive_swrt_start = time.time()
            
            try:
                adaptive_swrt_events = run_adaptive_swrt_analysis(wind_data_analysis, nominal)
                adaptive_swrt_events = add_turbine_id_if_missing(adaptive_swrt_events, "Adaptive SWRT")
                
                adaptive_swrt_time = time.time() - adaptive_swrt_start
                method_times['Adaptive SWRT'] = adaptive_swrt_time
                
                save_events_by_turbine(adaptive_swrt_events, 
                                     os.path.join(output_dir, 'adaptive_swrt_events.xlsx'),
                                     'Turbine')
                
                adaptive_swrt_total_events = len(adaptive_swrt_events)
                results_summary['Adaptive SWRT'] = {
                    'total_events': adaptive_swrt_total_events,
                    'time': adaptive_swrt_time,
                    'status': 'Success'
                }
                
                print(f"   ✅ Adaptive SWRT completed: {adaptive_swrt_total_events} events in {adaptive_swrt_time:.2f}s")
                
            except Exception as e:
                adaptive_swrt_time = time.time() - adaptive_swrt_start
                method_times['Adaptive SWRT'] = adaptive_swrt_time
                results_summary['Adaptive SWRT'] = {'status': f'Failed: {e}', 'time': adaptive_swrt_time}
                print(f"   ❌ Adaptive SWRT failed: {e}")
        else:
            print("\n⏭️  5-6. Adaptive methods skipped (not available)")

        # ====================================================================
        # GENERATE COMPARISON REPORT
        # ====================================================================
        total_time = time.time() - total_start_time
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE ANALYSIS RESULTS")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = []
        for method, result in results_summary.items():
            if result['status'] == 'Success':
                row = {
                    'Method': method,
                    'Total_Events': result['total_events'],
                    'Execution_Time_s': result['time'],
                    'Events_per_Second': result['total_events'] / result['time'] if result['time'] > 0 else 0,
                    'Status': result['status']
                }
                
                # Add RBA-specific metrics
                if 'traditional_events' in result:
                    row['Traditional_Events'] = result['traditional_events']
                    row['MCMC_Events'] = result['mcmc_events']
                    row['Traditional_Quality'] = result['trad_quality']
                    row['MCMC_Quality'] = result['mcmc_quality']
                
                comparison_data.append(row)
            else:
                comparison_data.append({
                    'Method': method,
                    'Total_Events': 0,
                    'Execution_Time_s': result['time'],
                    'Events_per_Second': 0,
                    'Status': result['status']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison report
        save_xls({'Method_Comparison': comparison_df}, 
                os.path.join(output_dir, 'method_comparison_report.xlsx'))
        
        # Print summary
        print(f"📋 Results saved to: {output_dir}")
        print("\n🏆 METHOD PERFORMANCE SUMMARY:")
        print("-" * 80)
        for _, row in comparison_df.iterrows():
            if row['Status'] == 'Success':
                print(f"{row['Method']:<20} | {row['Total_Events']:>6} events | {row['Execution_Time_s']:>6.2f}s | {row['Events_per_Second']:>8.2f} ev/s")
            else:
                print(f"{row['Method']:<20} | {'FAILED':<6} | {row['Execution_Time_s']:>6.2f}s | {row['Status']}")
        
        print(f"\n⏱️  TOTAL EXECUTION TIME: {total_time:.2f} seconds")
        print(f"📁 All results saved to: {output_dir}")
        print("🗂️  Each Excel file contains separate sheets for turbines")
        
        # Performance highlights
        successful_methods = comparison_df[comparison_df['Status'] == 'Success']
        if len(successful_methods) > 0:
            fastest_method = successful_methods.loc[successful_methods['Execution_Time_s'].idxmin()]
            most_events = successful_methods.loc[successful_methods['Total_Events'].idxmax()]
            
            print(f"\n🎯 PERFORMANCE HIGHLIGHTS:")
            print(f"   🚀 Fastest: {fastest_method['Method']} ({fastest_method['Execution_Time_s']:.2f}s)")
            print(f"   📊 Most events: {most_events['Method']} ({most_events['Total_Events']} events)")
        
        print("\n✅ Comprehensive analysis completed successfully!")
        
        return {
            'total_time': total_time,
            'results_summary': results_summary,
            'comparison_df': comparison_df,
            'output_dir': output_dir
        }
        
    except Exception as e:
        total_time = time.time() - total_start_time
        logger.error(f"Comprehensive analysis failed after {total_time:.2f} seconds: {e}")
        print(f"❌ Error after {total_time:.2f} seconds: {e}")
        raise


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(data_path, run_comprehensive=True, output_dir=None, interactive=True,
         selected_columns=None, analysis_type=None):
    """
    Enhanced main function with comprehensive multi-method analysis
    
    Args:
        data_path: Path to input data file
        run_comprehensive: Whether to run all methods
        output_dir: Custom output directory (optional)
        interactive: Whether to prompt for column selection
        selected_columns: Pre-selected columns (for non-interactive mode)
        analysis_type: Pre-selected analysis type (for non-interactive mode)
    
    Returns:
        Dictionary with results
    """
    main_start_time = time.time()
    logger.info("Starting Comprehensive Multi-Method Wind Turbine Analysis")
    logger.info(f"Data path: {data_path}")
    
    # Check if file exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = 'simulations/all_tests_together'
    
    try:
        if run_comprehensive:
            result = comprehensive_analysis(
                data_path, 
                use_optimization=True, 
                output_dir=output_dir,
                interactive=interactive,
                selected_columns=selected_columns,
                analysis_type=analysis_type
            )
        else:
            # Legacy single method execution
            result = comprehensive_analysis(
                data_path, 
                use_optimization=False, 
                output_dir=output_dir,
                interactive=interactive,
                selected_columns=selected_columns,
                analysis_type=analysis_type
            )
        
        main_total_time = time.time() - main_start_time
        print(f"\n⏱️  Total execution time: {main_total_time:.2f} seconds")
        return result
        
    except Exception as e:
        main_total_time = time.time() - main_start_time
        logger.error(f"Main execution failed after {main_total_time:.2f} seconds: {e}")
        raise


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comprehensive Wind Turbine Event Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for column selection)
  python event_detector.py --data input_data/Baltic_Eagle.xlsx
  
  # Non-interactive univariate (specify column)
  python event_detector.py --data input_data/Baltic_Eagle.xlsx --column Power
  
  # Non-interactive multivariate (specify multiple columns)
  python event_detector.py --data input_data/Baltic_Eagle.xlsx --columns Power,Windspeed
  
  # Custom output directory
  python event_detector.py --data input_data/Baltic_Eagle.xlsx --output results/
  
  # Quick analysis (no optimization)
  python event_detector.py --data input_data/Baltic_Eagle.xlsx --no-optimization
        """
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to input data (Excel file)')
    
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: simulations/all_tests_together)')
    
    parser.add_argument('--column', type=str, default=None,
                       help='Single column for univariate analysis (non-interactive mode)')
    
    parser.add_argument('--columns', type=str, default=None,
                       help='Comma-separated columns for multivariate analysis (non-interactive mode)')
    
    parser.add_argument('--no-optimization', action='store_true',
                       help='Skip parameter optimization for faster execution')
    
    parser.add_argument('--non-interactive', action='store_true',
                       help='Run in non-interactive mode (auto-select first column if not specified)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        sys.exit(1)
    
    # Determine if interactive mode
    interactive = not args.non_interactive
    
    # Parse column arguments for non-interactive mode
    selected_columns = None
    analysis_type = None
    
    if args.column and args.columns:
        print("❌ Error: Cannot specify both --column and --columns. Choose one.")
        sys.exit(1)
    
    if args.column:
        selected_columns = [args.column]
        analysis_type = 'univariate'
        interactive = False
        print(f"✓ Non-interactive mode: Univariate analysis on column '{args.column}'")
    
    elif args.columns:
        selected_columns = [col.strip() for col in args.columns.split(',')]
        analysis_type = 'multivariate'
        interactive = False
        print(f"✓ Non-interactive mode: Multivariate analysis on {len(selected_columns)} columns")
    
    # Run event detection
    try:
        result = main(
            data_path=args.data,
            run_comprehensive=not args.no_optimization,
            output_dir=args.output,
            interactive=interactive,
            selected_columns=selected_columns,
            analysis_type=analysis_type
        )
        
        print("\n" + "="*80)
        print("✅ SUCCESS")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
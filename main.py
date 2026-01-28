"""
main.py - Intelligent Event Detection and Prediction System
CLI-based workflow orchestrator for wind turbine and power system analysis

Workflow Architecture:
1. Event Detection: Traditional RBA-theta analysis (event_detector.py)
2. Event Prediction: ML-based forecasting with multiple approaches
   - Binary: SARIMAX-RBA (workflow1.py)
   - Multi-horizon: LSTM-based (workflow3.py) or Transformer-based (workflow4.py)
   - Transfer Learning: Pre-trained models (workflow3.py)
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('main_execution.log')
    ]
)
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """
    Intelligent dataset analyzer for determining data type and characteristics
    """
    
    def __init__(self):
        self.wind_keywords = [
            'wind', 'turbine', 'rotor', 'blade', 'nacelle', 'tower',
            'wind_speed', 'wind_power', 'wind_direction', 'pitch', 'yaw'
        ]
        
        self.electricity_keywords = [
            'price', 'electricity', 'load', 'demand', 'voltage', 'current',
            'frequency', 'power_grid', 'transmission', 'distribution'
        ]
        
        self.power_keywords = [
            'power', 'energy', 'generation', 'capacity', 'output',
            'mw', 'kw', 'megawatt', 'kilowatt'
        ]
    
    def load_dataset(self, path: str) -> Tuple[pd.DataFrame, bool]:
        """
        Load dataset with automatic format detection
        
        Returns:
            (DataFrame, success_flag)
        """
        try:
            logger.info(f"Loading dataset from: {path}")
            
            # Determine file extension
            file_ext = Path(path).suffix.lower()
            
            if file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(path)
            elif file_ext == '.csv':
                df = pd.read_csv(path)
            elif file_ext == '.parquet':
                df = pd.read_parquet(path)
            else:
                logger.error(f"Unsupported file format: {file_ext}")
                return None, False
            
            logger.info(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df, True
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None, False
    
    def detect_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect time/datetime column"""
        time_candidates = [
            'datetime', 'date', 'time', 'timestamp', 'dt',
            'DateTime', 'Date', 'Time', 'Timestamp'
        ]
        
        # Check exact matches
        for col in df.columns:
            if col in time_candidates:
                return col
        
        # Check partial matches
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'time']):
                return col
        
        # Check data types
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        
        return None
    
    def detect_dataset_type(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive dataset analysis
        
        Returns:
            {
                'type': 'wind' | 'electricity' | 'power' | 'unknown',
                'confidence': float (0-1),
                'time_column': str or None,
                'target_columns': list,
                'resolution': str (e.g., '10min', '1h'),
                'duration': str (e.g., '2 years'),
                'missing_data_pct': float
            }
        """
        logger.info("\n" + "="*80)
        logger.info("DATASET ANALYSIS")
        logger.info("="*80)
        
        analysis = {
            'type': 'unknown',
            'confidence': 0.0,
            'time_column': None,
            'target_columns': [],
            'resolution': None,
            'duration': None,
            'missing_data_pct': 0.0
        }
        
        # 1. Detect time column
        time_col = self.detect_time_column(df)
        analysis['time_column'] = time_col
        
        if time_col:
            try:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                time_diffs = df[time_col].diff().dropna()
                median_diff = time_diffs.median()
                
                # Determine resolution
                if median_diff < pd.Timedelta(minutes=1):
                    analysis['resolution'] = f"{int(median_diff.total_seconds())}sec"
                elif median_diff < pd.Timedelta(hours=1):
                    analysis['resolution'] = f"{int(median_diff.total_seconds() / 60)}min"
                else:
                    analysis['resolution'] = f"{int(median_diff.total_seconds() / 3600)}h"
                
                # Calculate duration
                duration = df[time_col].max() - df[time_col].min()
                if duration.days < 30:
                    analysis['duration'] = f"{duration.days} days"
                elif duration.days < 365:
                    analysis['duration'] = f"{duration.days / 30:.1f} months"
                else:
                    analysis['duration'] = f"{duration.days / 365:.1f} years"
                
                logger.info(f"✓ Time column detected: {time_col}")
                logger.info(f"  Resolution: {analysis['resolution']}")
                logger.info(f"  Duration: {analysis['duration']}")
            except Exception as e:
                logger.warning(f"Could not parse time column: {e}")
        
        # 2. Analyze column names for keywords
        columns_lower = [col.lower() for col in df.columns]
        columns_text = ' '.join(columns_lower)
        
        wind_score = sum(1 for kw in self.wind_keywords if kw in columns_text)
        electricity_score = sum(1 for kw in self.electricity_keywords if kw in columns_text)
        power_score = sum(1 for kw in self.power_keywords if kw in columns_text)
        
        # 3. Detect target columns (numeric columns likely to be targets)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_col and time_col in numeric_cols:
            numeric_cols.remove(time_col)
        
        # Prioritize columns with power/energy keywords
        target_candidates = []
        for col in numeric_cols:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['power', 'energy', 'output', 'generation']):
                target_candidates.append(col)
        
        if not target_candidates:
            # Fallback: use all numeric columns
            target_candidates = numeric_cols[:5]  # Limit to first 5
        
        analysis['target_columns'] = target_candidates
        
        # 4. Determine dataset type
        total_score = wind_score + electricity_score + power_score
        
        if wind_score > max(electricity_score, power_score):
            analysis['type'] = 'wind'
            analysis['confidence'] = wind_score / max(total_score, 1)
        elif electricity_score > max(wind_score, power_score):
            analysis['type'] = 'electricity'
            analysis['confidence'] = electricity_score / max(total_score, 1)
        elif power_score > 0:
            analysis['type'] = 'power'
            analysis['confidence'] = power_score / max(total_score, 1)
        else:
            analysis['type'] = 'unknown'
            analysis['confidence'] = 0.0
        
        # 5. Missing data analysis
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        analysis['missing_data_pct'] = missing_pct
        
        # Print analysis summary
        logger.info(f"\nDataset Type: {analysis['type'].upper()}")
        logger.info(f"Confidence: {analysis['confidence']*100:.1f}%")
        logger.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        logger.info(f"Target columns detected: {len(analysis['target_columns'])}")
        if analysis['target_columns']:
            logger.info(f"  Primary targets: {analysis['target_columns'][:3]}")
        logger.info(f"Missing data: {missing_pct:.2f}%")
        logger.info("="*80 + "\n")
        
        return analysis


class WorkflowOrchestrator:
    """
    Main workflow orchestrator for event detection and prediction
    """
    
    def __init__(self):
        self.analyzer = DatasetAnalyzer()
        
        # Define workflow paths (support both flat and nested structure)
        self.workflows = {
            'event_detector': self._find_workflow('event_detector.py'),
            'workflow1': self._find_workflow('workflow1.py'),
            'workflow3': self._find_workflow('workflow3.py'),
            'workflow4': self._find_workflow('workflow4.py')
        }
        
        # Verify workflow files exist
        self._verify_workflows()
    
    def _find_workflow(self, filename: str) -> str:
        """
        Find workflow file in multiple possible locations
        
        Priority:
        1. workflows/ subdirectory
        2. Same directory as main.py
        3. Current working directory
        """
        # Check workflows/ subdirectory first
        workflows_dir = os.path.join(os.path.dirname(__file__), 'workflows', filename)
        if os.path.exists(workflows_dir):
            return workflows_dir
        
        # Check same directory as main.py
        same_dir = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(same_dir):
            return same_dir
        
        # Check current working directory
        if os.path.exists(filename):
            return filename
        
        # Return relative path (will be checked in _verify_workflows)
        return os.path.join('workflows', filename)
    
    def _verify_workflows(self):
        """Check if workflow scripts exist"""
        missing = []
        available = []
        
        for name, filepath in self.workflows.items():
            if filepath and os.path.exists(filepath):
                available.append(f"{name} ({filepath})")
            else:
                missing.append(f"{name} ({filepath})")
        
        if available:
            logger.info(f"✓ Available workflows:")
            for wf in available:
                logger.info(f"  - {wf}")
        
        if missing:
            logger.warning(f"⚠️  Missing workflows:")
            for wf in missing:
                logger.warning(f"  - {wf}")
            logger.warning("Some features may not be available")
    
    def run_workflow(self, workflow_name: str, data_path: str, **kwargs) -> bool:
        """
        Execute a specific prediction workflow
        """
        logger.info("\n" + "="*80)
        logger.info(f"RUNNING {workflow_name.upper()}")
        logger.info("="*80)
        
        workflow_file = self.workflows.get(workflow_name)
        if not workflow_file or not os.path.exists(workflow_file):
            logger.error(f"Workflow not found: {workflow_name}")
            logger.error(f"Expected location: {workflow_file}")
            return False
        
        try:
            start_time = time.time()
            
            # Add workflows directory to path if it exists
            workflows_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workflows')
            if os.path.exists(workflows_dir) and workflows_dir not in sys.path:
                sys.path.insert(0, workflows_dir)
            
            # Also add parent directory to path
            parent_dir = os.path.dirname(os.path.abspath(__file__))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Dynamic import with better error handling
            result = None
            
            if workflow_name == 'workflow3':
                # LSTM-RBA
                logger.info("Launching LSTM-RBA (Multi-Horizon Event-Aware Prediction)")
                
                use_transfer_learning = kwargs.get('transfer_learning', False)
                if use_transfer_learning:
                    logger.info("Using transfer learning mode")
                
                # Try multiple import strategies
                try:
                    # Strategy 1: Import from workflows package
                    from workflows.workflow3 import main as workflow3_main
                    result = workflow3_main(
                        data_path=data_path,
                        transfer_learning=use_transfer_learning,
                        config_overrides=kwargs.get('config_overrides')
                    )
                except ImportError as e1:
                    logger.warning(f"Could not import from workflows package: {e1}")
                    try:
                        # Strategy 2: Direct import
                        import workflow3
                        result = workflow3.main(
                            data_path=data_path,
                            transfer_learning=use_transfer_learning,
                            config_overrides=kwargs.get('config_overrides')
                        )
                    except ImportError as e2:
                        logger.error(f"Could not import workflow3 at all: {e2}")
                        raise
            
            # Similar logic for workflow1 and workflow4...
            elif workflow_name == 'workflow1':
                logger.info("Launching SARIMAX-RBA (Binary Prediction)")
                try:
                    from workflows.workflow1 import main as workflow1_main
                    result = workflow1_main(data_path=data_path, **kwargs)
                except ImportError:
                    import workflow1
                    result = workflow1.main(data_path=data_path, **kwargs)
            
            elif workflow_name == 'workflow4':
                logger.info("Launching Transformer-RBA (High Generalization)")
                try:
                    from workflows.workflow4 import main as workflow4_main
                    result = workflow4_main(
                        data_path=data_path,
                        transfer_learning=kwargs.get('transfer_learning', False),
                        config_overrides=kwargs.get('config_overrides')
                    )
                except ImportError:
                    import workflow4
                    result = workflow4.main(
                        data_path=data_path,
                        transfer_learning=kwargs.get('transfer_learning', False),
                        config_overrides=kwargs.get('config_overrides')
                    )
            
            execution_time = time.time() - start_time
            logger.info(f"✓ {workflow_name} completed in {execution_time:.2f} seconds")
            
            # Print result summary
            if result and isinstance(result, dict):
                if result.get('mode') == 'transfer_learning':
                    logger.info("Transfer learning inference completed")
                    logger.info(f"Predictions saved to: {result.get('model_dir', 'N/A')}")
                else:
                    logger.info("Full training completed")
                    logger.info(f"Results saved to: {result.get('output_dir', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"{workflow_name} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_event_detection(self, data_path: str) -> bool:
        """
        Execute event detection workflow (event_detector.py)
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING EVENT DETECTION WORKFLOW")
        logger.info("="*80)
        
        detector_path = self.workflows.get('event_detector')
        if not detector_path or not os.path.exists(detector_path):
            logger.error(f"Event detector not found: {detector_path}")
            return False
        
        try:
            # Add parent directory to path
            parent_dir = os.path.dirname(os.path.abspath(detector_path))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Import and run event_detector
            import event_detector
            
            logger.info(f"Launching event detection on: {data_path}")
            start_time = time.time()
            
            result = event_detector.main(data_path, run_comprehensive=True)
            
            execution_time = time.time() - start_time
            logger.info(f"✓ Event detection completed in {execution_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Event detection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def interactive_menu(self):
        """
        Interactive CLI menu for user interaction
        """
        print("\n" + "="*80)
        print("🌪️  INTELLIGENT EVENT DETECTION & PREDICTION SYSTEM")
        print("="*80)
        print("\nWelcome! This system provides:")
        print("  1. Traditional event detection (RBA-theta)")
        print("  2. ML-based event prediction (SARIMAX, LSTM, Transformer integrated with RBA-theta)")
        print("="*80 + "\n")
        
        # STEP 1: Choose main mode
        print("STEP 1: Select Mode")
        print("-" * 40)
        print("1. Event Detection (Analyze historical events)")
        print("2. Event Prediction (Forecast future events)")
        
        mode_choice = self._get_user_input("Enter your choice (1 or 2): ", ['1', '2'])
        
        # STEP 2: Get dataset path
        data_path = self._get_dataset_path()
        if not data_path:
            logger.error("Invalid dataset path. Exiting.")
            return
        
        # STEP 3: Execute based on mode
        if mode_choice == '1':
            # EVENT DETECTION
            self.run_event_detection(data_path)
        
        else:
            # EVENT PREDICTION
            self._event_prediction_workflow(data_path)
    
    def _event_prediction_workflow(self, data_path: str):
        """
        Event prediction workflow with intelligent routing
        """
        # Analyze dataset
        df, success = self.analyzer.load_dataset(data_path)
        if not success:
            logger.error("Failed to load dataset. Exiting.")
            return
        
        analysis = self.analyzer.detect_dataset_type(df)
        
        # Check if wind-related
        if analysis['type'] != 'wind':
            print("\n" + "="*80)
            print("NON-WIND DATASET DETECTED")
            print("="*80)
            print(f"Dataset type: {analysis['type']}")
            print(f"Confidence: {analysis['confidence']*100:.1f}%")
            print("\nFor non-wind power data, using LSTM-RBA workflow (workflow3.py)")
            print("This is the most versatile approach for general time-series prediction.")
            
            # Always use workflow3 for non-wind data
            self.run_workflow('workflow3', data_path)
            return
        
        # WIND DATASET - Continue with detailed options
        print("\n" + "="*80)
        print("WIND DATASET DETECTED")
        print("="*80)
        print(f"Confidence: {analysis['confidence']*100:.1f}%")
        print(f"Time resolution: {analysis['resolution']}")
        print(f"Duration: {analysis['duration']}")
        
        # STEP 3: Prediction type
        print("\n" + "-"*80)
        print("STEP 2: Select Prediction Type")
        print("-" * 40)
        print("1. Binary Prediction (Event occurrence: Yes/No)")
        print("2. Multi-Horizon Event-Aware Prediction (When, Where, How severe)")
        
        pred_type = self._get_user_input("Enter your choice (1 or 2): ", ['1', '2'])
        
        if pred_type == '1':
            # BINARY PREDICTION -> SARIMAX-RBA
            print("\n✓ Selected: Binary Prediction")
            print("Using SARIMAX-RBA workflow (workflow1.py)")
            self.run_workflow('workflow1', data_path)
        
        else:
            # MULTI-HORIZON PREDICTION
            print("\n✓ Selected: Multi-Horizon Event-Aware Prediction")
            
            # STEP 4: Precision vs Generalization
            print("\n" + "-"*80)
            print("STEP 3: Model Characteristics")
            print("-" * 40)
            print("1. High Precision (Best for similar conditions, detailed predictions)")
            print("   → LSTM-RBA: Superior for well-defined patterns")
            print("\n2. High Generalization (Best for diverse conditions, robustness)")
            print("   → Transformer-RBA: Better for varied scenarios")
            
            model_type = self._get_user_input("Enter your choice (1 or 2): ", ['1', '2'])
            
            # STEP 5: Training mode
            print("\n" + "-"*80)
            print("STEP 4: Training Mode")
            print("-" * 40)
            print("1. Full Retraining (Train from scratch on your data)")
            print("   → Best performance, but slower")
            print("\n2. Transfer Learning (Use pre-trained model + fine-tuning)")
            print("   → Faster, good for limited data")
            
            training_mode = self._get_user_input("Enter your choice (1 or 2): ", ['1', '2'])
            
            # Execute workflow based on choices
            if training_mode == '1':
                # FULL RETRAINING
                if model_type == '1':
                    # High Precision -> workflow3 (LSTM)
                    print("\n✓ Configuration: High Precision + Full Retraining")
                    print("Using LSTM-RBA workflow (workflow3.py)")
                    self.run_workflow('workflow3', data_path, transfer_learning=False)
                else:
                    # High Generalization -> workflow4 (Transformer)
                    print("\n✓ Configuration: High Generalization + Full Retraining")
                    print("Using Transformer-RBA workflow (workflow4.py)")
                    self.run_workflow('workflow4', data_path, transfer_learning=False)
            
            else:
                # TRANSFER LEARNING
                # Always use workflow3 for transfer learning (both precision types)
                print("\n✓ Configuration: Transfer Learning")
                print("Using LSTM-RBA workflow (workflow3.py) with transfer learning")
                print("(Transfer learning implementation supports both precision types)")
                self.run_workflow('workflow3', data_path, transfer_learning=True)
    
    def _get_user_input(self, prompt: str, valid_choices: list) -> str:
        """Get validated user input"""
        while True:
            try:
                choice = input(prompt).strip()
                if choice in valid_choices:
                    return choice
                print(f"Invalid choice. Please enter one of: {', '.join(valid_choices)}")
            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user.")
                sys.exit(0)
            except Exception as e:
                print(f"Error: {e}")
    
    def _get_dataset_path(self) -> Optional[str]:
        """Get and validate dataset path from user"""
        print("\n" + "-"*80)
        print("Dataset Input")
        print("-" * 40)
        
        while True:
            try:
                path = input("Enter dataset path (or 'q' to quit): ").strip()
                
                if path.lower() == 'q':
                    print("Exiting...")
                    return None
                
                # Remove quotes if present
                path = path.strip('"').strip("'")
                
                # Check if file exists
                if os.path.exists(path):
                    logger.info(f"✓ Dataset found: {path}")
                    return path
                else:
                    print(f"❌ File not found: {path}")
                    print("Please check the path and try again.\n")
            
            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user.")
                return None
            except Exception as e:
                print(f"Error: {e}")


def parse_arguments():
    """
    Parse command-line arguments for non-interactive mode
    """
    parser = argparse.ArgumentParser(
        description='Intelligent Event Detection & Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py
  
  # Event detection
  python main.py --mode detection --data path/to/data.xlsx
  
  # Binary prediction
  python main.py --mode prediction --pred-type binary --data path/to/data.xlsx
  
  # Multi-horizon with LSTM (full training)
  python main.py --mode prediction --pred-type multi-horizon --model lstm --data path/to/data.xlsx
  
  # Multi-horizon with Transformer (transfer learning)
  python main.py --mode prediction --pred-type multi-horizon --model transformer --transfer --data path/to/data.xlsx
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['detection', 'prediction'],
        help='Operation mode: detection or prediction'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to dataset file'
    )
    
    parser.add_argument(
        '--pred-type',
        choices=['binary', 'multi-horizon'],
        help='Prediction type (only for prediction mode)'
    )
    
    parser.add_argument(
        '--model',
        choices=['lstm', 'transformer'],
        help='Model type for multi-horizon prediction'
    )
    
    parser.add_argument(
        '--transfer',
        action='store_true',
        help='Use transfer learning (only for multi-horizon)'
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Create orchestrator
        orchestrator = WorkflowOrchestrator()
        
        # Check if running in interactive or CLI mode
        if args.mode is None:
            # INTERACTIVE MODE
            orchestrator.interactive_menu()
        
        else:
            # NON-INTERACTIVE MODE (CLI)
            if not args.data:
                print("Error: --data argument is required in non-interactive mode")
                sys.exit(1)
            
            # Validate dataset path
            if not os.path.exists(args.data):
                print(f"Error: Dataset not found: {args.data}")
                sys.exit(1)
            
            # Execute based on mode
            if args.mode == 'detection':
                orchestrator.run_event_detection(args.data)
            
            elif args.mode == 'prediction':
                if not args.pred_type:
                    print("Error: --pred-type required for prediction mode")
                    sys.exit(1)
                
                # Analyze dataset first
                df, success = orchestrator.analyzer.load_dataset(args.data)
                if not success:
                    sys.exit(1)
                
                analysis = orchestrator.analyzer.detect_dataset_type(df)
                
                # Route to appropriate workflow
                if args.pred_type == 'binary':
                    # SARIMAX-RBA
                    orchestrator.run_workflow('workflow1', args.data)
                
                elif args.pred_type == 'multi-horizon':
                    if analysis['type'] != 'wind':
                        # Non-wind -> always workflow3
                        print(f"Non-wind dataset detected. Using LSTM-RBA (workflow3)")
                        orchestrator.run_workflow('workflow3', args.data)
                    else:
                        # Wind dataset
                        if args.transfer:
                            # Transfer learning -> workflow3
                            orchestrator.run_workflow('workflow3', args.data, transfer_learning=True)
                        else:
                            # Full training
                            if args.model == 'transformer':
                                orchestrator.run_workflow('workflow4', args.data)
                            else:
                                # Default to LSTM
                                orchestrator.run_workflow('workflow3', args.data)
        
        print("\n" + "="*80)
        print("✅ EXECUTION COMPLETED")
        print("="*80)
        print("Check the output directories for results.")
        print("Logs saved to: main_execution.log")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
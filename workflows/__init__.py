# workflows/__init__.py
"""
Workflows package for event prediction models
"""

# Make workflows easily importable
try:
    from .workflow3 import main as workflow3_main, run_inference_only as workflow3_inference
    __all__ = ['workflow3_main', 'workflow3_inference']
except ImportError as e:
    print(f"Warning: Could not import workflow3: {e}")
    __all__ = []

# Add workflow1 and workflow4 when available
try:
    from .workflow1 import main as workflow1_main
    __all__.append('workflow1_main')
except ImportError:
    pass

try:
    from .workflow4 import main as workflow4_main
    __all__.append('workflow4_main')
except ImportError:
    pass
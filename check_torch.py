import sys
print(f"Python version: {sys.version}")
try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"Torch file: {torch.__file__}")
    from torch import Tensor
    print("Successfully imported Tensor from torch")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

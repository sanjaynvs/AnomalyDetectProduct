try:
    from torch.utils.data import DataLoader
    print("DataLoader is available.")
    print(f"Type: {type(DataLoader)}")
except ImportError:
    print("DataLoader is not available. Please install PyTorch.")

import torch
print(torch.__version__)
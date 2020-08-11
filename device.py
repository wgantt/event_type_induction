from torch import device, cuda

DEFAULT_DEVICE = device('cuda' if cuda.is_available() else 'cpu')

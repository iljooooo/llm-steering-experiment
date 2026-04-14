'''Container of deterministic configuration variables'''
from pathlib import Path
from torch.mps import is_available as mps_is_available
from torch.cuda import is_available as cuda_is_available

def device() -> str:
    '''Standard method to identify which device you are supposed to use for GPU processing'''
    if cuda_is_available():
        return 'cuda'
    elif mps_is_available():
        return 'mps'
    else:
        return 'cpu'
##

ROOT_PATH = Path(__file__).parent.parent # project root path
DEVICE = device()

MODELS = {
    'llama3.1': 'meta-llama/Llama-3.1-8B-Instruct'
}
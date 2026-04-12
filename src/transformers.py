import json
import os
import subprocess
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import Module
from typing import Tuple, Any, Literal

_Device = Literal['cpu', 'cuda', 'mps']

'''
Loader methods for HuggingFace transformers models. This module contains also API support and is intended to be self-contained
'''

#TODO: add support for other types of models (e.g. BERT, T5, etc.)
#TODO: define a 'Tokenizer' class with common utils.
#TODO: authomatize API key management (e.g. load from file, environment variable, etc.)

# huggingface API key should be available at:
# ~/.cache/huggingface/token

def load_model(model_name, device: _Device = 'cpu') -> Module:
    model =  AutoModelForCausalLM.from_pretrained(
        model_name,
        # token = API_KEY,
        device_map=device,
        torch_dtype='auto'
    )

    model.eval() # avoids dropouts
    return model
##

def load_tokenizer(model_name) -> Any:
    return AutoTokenizer.from_pretrained(
        model_name,
    )
##

def load_all(model_name) -> Tuple[Module, Any]: 
    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)

    return model, tokenizer
##
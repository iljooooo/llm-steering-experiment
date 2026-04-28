'''We use this script to compute the steering vectors for all different areas using llama3.1 as our base model'''

## ALLOWING IMPORTATION FROM THE PARENT DIR ##
from collections import defaultdict
import sys
import json
from pathlib import Path
from huggingface_hub import logging
from typing import Literal, Union

import torch
import torch.mps as mps

PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

import src
logging.set_verbosity_debug()
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


from config import _DatasetType, _AnthropicDatasetType
def load_dataset(
        ds: _DatasetType,
        concept: _AnthropicDatasetType # might be changed in the future
    ) -> dict:
    TRAINING_DS_PATH = PROJECT_DIR / 'datasets' / ds / 'train' / f'{concept}.json'

    # might include metadata for specific datasets?
    with open(TRAINING_DS_PATH) as f:
        data = json.load(f)
    return data
##


if __name__ == '__main__':
    print(PROJECT_DIR)
    llama_model, llama_tokenizer = src.load_all(MODEL_NAME)
    llama_model.to('mps')
    llama_model.eval()

    extracting = (
    'coordinate-other-ais',
    'corrigible-neutral-HHH',
    'hallucination',
    'myopic-reward',
    'refusal',
    'survival-instinct',
    'sycophancy'
    ) # set to tuple for immutability (and so type checkers coherency)

    llama_shape = (33, 4096) # shape 
    steering_vecs = {}

    print('='*5 + f'EXTRACTING ANTHROPIC STEERING VECTORS' + '='*5)
    data = {}
    for key in extracting:

        STEERING_VECTORS_PATH = PROJECT_DIR / 'steering-vectors' / 'meta-llama' / 'Llama-3.1-8B-Instruct' / f'{key}.pt'

        if STEERING_VECTORS_PATH.exists():
            print(f'Steering vector for {STEERING_VECTORS_PATH} already exists! Moving on to the next category')
            continue

        print('='*5 + f'USING \'{key}\' DATASET' + '='*5)
        steering_vecs[key] = torch.zeros(llama_shape, device='cpu')
        data[key] = load_dataset('anthropic', key)

        for i, prompt in enumerate(data[key]):
            if i<10:
                print(f'Evaluating prompt n° {i+1}, total occupied memory: {mps.current_allocated_memory() / (1024**2)} MB')
            if (i+1) % 50 == 0:
                print(f'Evaluating prompt n° {i+1}')

            question = prompt['question']
            answer_matching = prompt['answer_matching_behavior']
            answer_not_matching = prompt['answer_not_matching_behavior']

            prompt_matching = question + answer_matching
            prompt_not_matching = question + answer_not_matching

            ## TOKENIZATION
            inputs_matching = llama_tokenizer(
                prompt_matching,
                return_tensors='pt'
            )
            inputs_not_matching = llama_tokenizer(
                prompt_not_matching,
                return_tensors='pt'
            )

            ## MOVING TO MPS
            inputs_matching = {k:v.to('mps') for k,v in inputs_matching.items()}
            inputs_not_matching = {k:v.to('mps') for k,v in inputs_not_matching.items()}

            ## FORWARD COMPUTATION
            with torch.inference_mode():
                outs_matching = llama_model(
                    **inputs_matching,
                    output_hidden_states=True        
                )
                outs_not_matching = llama_model(
                    **inputs_not_matching,
                    output_hidden_states=True
                )

            ## GETTING THE STEERS AND MOVING THEM TO CPU
            # TODO: shall we refer to [-3] element instead? Possibly because the logits are always predictive.
            positive_steer = torch.stack([hs.squeeze()[-2] for hs in outs_matching.hidden_states]).cpu()
            negative_steer = torch.stack([hs.squeeze()[-2] for hs in outs_not_matching.hidden_states]).cpu()

            ## UPDATING THE STEERING VECTORS
            steering_vecs[key] *= i
            steering_vecs[key] += (positive_steer - negative_steer)
            steering_vecs[key] /= (i+1)

            ## MOVING BACK TO CPU
            inputs_matching = {k:v.to('cpu') for k,v in inputs_matching.items()}
            inputs_not_matching = {k:v.to('cpu') for k,v in inputs_not_matching.items()}

            ## EMPTYING THE CACHE
            del inputs_matching, inputs_not_matching, outs_matching, outs_not_matching, positive_steer, negative_steer
            mps.empty_cache()
        ## end of computation

        print('Finished the computation for prompts in \'{key}\' cathegory')

        ## DATA LOADING IN APPROPRIATE FILES
        torch.save(steering_vecs[key], STEERING_VECTORS_PATH)
        print(f'[INFO] Saved obtained steetring vectors in {STEERING_VECTORS_PATH}')

    ## end of ALL computation

    print('[INFO] All steering vectors are computed and saved!')
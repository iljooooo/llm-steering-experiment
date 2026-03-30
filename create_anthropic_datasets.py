'''Randomly samples from the original datasets in order to have a bound on the numerical values of the DS size. All this values are declared at the beginning of the script and editable. Some of the general location(s) are extracted from the `src/config` file, even though this file can ideally be seen also as a configurator for the whole databases.'''

import random as rd
import json
import warnings
import requests

from src.config import ROOT_PATH

## DECLARATION, FEEL FREE TO CHANGE ##
MAX_ROWS = 1000
TEST_ROWS = 50
SEED = 1234
rd.seed(SEED) # needed when shuffling big data

## DATABASE EXTRACTION ##
'''
GENERAL IDEA:

Raw dataset(s) available at:
    -> ROOT_PATH / datasets / anthropic / raw

Extracted datasets available at:
    -> ROOT_PATH / datasets / anthropic / train
    -> ROOT_PATH / datasets / anthropic / test
'''

ANTHROPIC_FOLDER_PATH = ROOT_PATH / 'datasets' / 'anthropic'
if not ANTHROPIC_FOLDER_PATH.exists():
    raise FileNotFoundError(f'Unable to locate the designed folder at {ANTHROPIC_FOLDER_PATH}')


ANTHROPIC_DATASET_CURL = {
    'sycophancy': 'https://raw.githubusercontent.com/nrimsky/CAA/refs/heads/main/datasets/raw/sycophancy/dataset.json',
    'survival-instinct': 'https://raw.githubusercontent.com/nrimsky/CAA/refs/heads/main/datasets/raw/survival-instinct/dataset.json',
    'refusal': 'https://raw.githubusercontent.com/nrimsky/CAA/refs/heads/main/datasets/raw/refusal/dataset.json',
    'myopic-reward': 'https://raw.githubusercontent.com/nrimsky/CAA/refs/heads/main/datasets/raw/myopic-reward/dataset.json',
    'hallucination': 'https://raw.githubusercontent.com/nrimsky/CAA/refs/heads/main/datasets/raw/hallucination/dataset.json',
    'corrigible-neutral-HHH': 'https://raw.githubusercontent.com/nrimsky/CAA/refs/heads/main/datasets/raw/corrigible-neutral-HHH/dataset.json',
    'coordinate-other-ais': 'https://raw.githubusercontent.com/nrimsky/CAA/refs/heads/main/datasets/raw/coordinate-other-ais/dataset.json',
}


# TODO: implement this (possibly via shell arguments parsing)
def _download_anthropic_datasets() -> None:
    '''Opportunely downloads the corresponding datasets for the Anthropic experiment. Uses components from ANTHROPIC_DATASET_CURL'''

    for ds, url in ANTHROPIC_DATASET_CURL.items():
        out_path = ANTHROPIC_FOLDER_PATH / 'raw' / (ds+'.json')

        if out_path.exists():
            print(f'[INFO]: {out_path} was already downloaded')
            continue # nothing to download, skip to next ds
        
        # else:
        print(f'[INFO]: Downloading {url}')

        response = requests.get(url)
        response.raise_for_status()

        out_path.write_bytes(response.content)
        print(f'[INFO]: saved ds {ds}.json to {out_path}\n')
    ##
##

def main():
    for file in (ANTHROPIC_FOLDER_PATH / 'raw').glob('*'):

        with open(file, 'r') as f:
            ds = json.load(f)
        print(file.name)
        
        
        # extracting test labels
        try:
            test_data = [ds.pop(rd.randrange(len(ds))) for _ in range(TEST_ROWS)]
        except ValueError:
            raise ValueError(f'Sample larger than population or is negative. Consider fixing by adjusting TEST_ROWS, currently {TEST_ROWS}')
        
        # extracting training labels
        if len(ds) <= MAX_ROWS:
            train_data = ds
        else:
            train_data = [ds.pop(rd.randrange(len(ds))) for _ in range(MAX_ROWS)]

        # saving
        if not (ANTHROPIC_FOLDER_PATH / 'test').exists():
            (ANTHROPIC_FOLDER_PATH / 'test').mkdir(parents=True, exist_ok=False)
        if not (ANTHROPIC_FOLDER_PATH / 'train').exists():
            (ANTHROPIC_FOLDER_PATH / 'train').mkdir(parents=True, exist_ok=False)

        with open(ANTHROPIC_FOLDER_PATH / 'test' / file.name, 'w', encoding='utf-8') as test_file:
            json.dump(test_data, test_file, ensure_ascii=False, indent=4)
        with open(ANTHROPIC_FOLDER_PATH / 'train' / file.name, 'w', encoding='utf-8') as train_file:
            json.dump(train_data, train_file, ensure_ascii=False, indent=4)
    ##
##
    

# DEBUG MODE
if __name__ == '__main__':
    _download_anthropic_datasets()
    main()
'''Contains constrains about currently implemented Datasets that allow for an easier control from typecheckers'''
from typing import Literal

_DatasetType = Literal[
    'anthropic'
]

_AnthropicDatasetType = Literal[
    'coordinate-other-ais',
    'corrigible-neutral-HHH',
    'hallucination',
    'myopic-reward',
    'refusal',
    'survival-instinct',
    'sycophancy'
]
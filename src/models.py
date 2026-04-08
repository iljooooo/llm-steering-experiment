'''
models.py aims to both abstract and examples wrapper to any model that we aim to support as an available option for the experimentation.
'''

from abc import abstractmethod

from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from typing import Any, Optional, Literal, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import LlamaForCausalLM


class _BaseModel(Module):
    '''
    Base class for supported modules. It should work on a 2-way basis:
    - introduce mandatory methods (abstract)
    - complete some missing common pipelines from `torch.nn.Module`, which is a parents class
    
    Abstract methods that mostly contains operations that include:
    - hooks assignment
    - setting/management of debug mode
    - general and common parameters
    - complete pipelines for specific experiments (computing steering vec, running tests...)
    '''

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._handles: Optional[List[RemovableHandle]] = None # contains info about the hooks
    ##    

    ## FORWARD_HOOKS
    @property
    def forward_hooks(self):
        pass
    @forward_hooks.getter
    def forward_hooks(self):
        pass
    @forward_hooks.setter
    def forward_hooks(self, value):
        pass
    ##

    ## PRE_FORWARD_HOOKS
    @property
    def pre_forward_hooks(self):
        pass
    @pre_forward_hooks.getter
    def pre_forward_hooks(self):
        pass
    @pre_forward_hooks.setter
    def pre_forward_hooks(self, value):
        pass
    ##



    def _remove_hooks(self) -> None:
        '''
        `torch` has an awkward method to handle hooks from (and for) a model. However the method is lacking some sort of consistency. For example, one can:
        - access either forward or pre-forward hooks proper of just a module (not of children)
        - access the full_backwards and backwards hooks of all the modules tree 

        Because of this unconsistent behaviour, I decided to fix it and to allow for global handling of such objects.

        This must be carefully used since it does not synchronize with currently used techniques.
        '''
        if self._handles is None:
            return
        
        [handle.remove for handle in self._handles]
        self._handles = None
    ##
##

class LLamaModel(_BaseModel, LlamaForCausalLM):
    pass

    ## CHECK OUT FOR LLamaForCauseLM.forward() method, since it allows for `logits_to_keep` argument
##
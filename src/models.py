'''
models.py aims to both abstract and examples wrapper to any model that we aim to support as an available option for the experimentation.
'''

from abc import abstractmethod

from torch import Value
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from typing import Any, Optional, List, Literal, Sequence, Union
from transformers import AutoModelForCausalLM
# from transformers.models.auto.modeling_auto import _BaseModelWithGenerate
# from transformers.models.llama import LlamaForCausalLM

class NonInstantiableModel(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
##

_DeviceType = Union[
    Literal['cpu'],
    Literal['mps'],
    Literal['cuda'],
    Literal['auto'] # auto is meant to represent cases in which the model has to switch between different devices
]

class _BaseModel:
    '''
    Base class for supported modules. It should work on a 2-way basis:
    - introduce mandatory methods (abstract)
    - complete some missing common pipelines from `torch.nn.Module`, which is a parents class
    
    Abstract methods that mostly contains operations that include:
    - hooks assignment
    - setting/management of debug mode
    - general and common parameters
    - complete pipelines for specific experiments (computing steering vec, running tests...)

    Currently does not support *args support for internal logic purposes
    '''

    def __init__(self, *args: Any, **kwargs: Any) -> None:

        self._model_name: Optional[str]


        if self._model_name is not None:
            self = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self._model_name, *args, **kwargs)
        else:
            raise NonInstantiableModel('Unable to recover model name from object metadata')
    
        self._handles: Optional[Sequence[RemovableHandle]] = None   # contains all the hooks info about the hooks
        self._device: _DeviceType = kwargs.get('device_map')        # contains info about the device

        if self._device == 'auto':
            pass #instantiate auto memory hooks!
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

    ## DIFFERENT METHODS THAT SHOULD BE INCLUDED: SPECIFIC HOOK ASSIGNMENT TO HANDLE DIFFERENT PHASES DURING THE EXPERIMENT
    @abstractmethod
    def _mode_to_steering_vector_computing(self):
        pass
    @abstractmethod
    def _mode_to_steering_closed_amswer(self):
        pass
    @abstractmethod
    def _mode_to_steered_open_generation(self):
        pass
    ## others? 

    def _remove_hooks(self) -> None:
        '''
        `torch` has an awkward method to handle hooks from (and for) a model. However the method is lacking some sort of consistency. For example, one can:
        - access either forward or pre-forward hooks proper of just a module (not of children)
        - access the full_backwards and backwards hooks of all the modules tree 

        Because of this unconsistent behaviour, I decided to fix it and to allow for global handling of such objects.

        This must be carefully used since it does not synchronize with currently used techniques.
        '''
        if self._handles is None:
            return # if we have no hooks there is no reasons to keep it here
        
        [handle.remove() for handle in self._handles]
        self._handles = None
    ##

    def _remove_pre_forward_hooks(self):
        pass
    def _remove_forward_hooks(self):
        pass
    def _remove_backwards_hooks(self):      # this is included even though backward hooks should not be a thing here
        pass
    ##
##

from transformers.models.llama.modeling_llama import LlamaForCausalLM
class LLamaModel(_BaseModel, LlamaForCausalLM):
    
    def __init__(
            self,
            device_type,
            *args,
            **kwargs
    ) -> None:
        '''Instantiate a wrapped LLamaModel following standards from the abstract class'''
        self._model_name = "meta-llama/Llama-3.1-8B-Instruct"
        super().__init__(*args, **kwargs)
    ##

    ## CHECK OUT FOR LLamaForCauseLM.forward() method, since it allows for `logits_to_keep` argument
##
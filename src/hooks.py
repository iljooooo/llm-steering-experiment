'''
Hooks are designed to handle debug functionalities of different modules, and to use the steered vectors obtained via previous computation.

They will be imported and used by `models.py` (or similar) in order to be automatically assigned and handled by built-in functionalities. See that (or those) files for more details.

Even though they are still not written, it is possible that we will write hooks also for training phase (extraction of steering vectors)
'''

import torch.mps as mps
import torch.cuda as cuda
from torch import Tensor, allclose
from torch.nn import Module
from typing import Any, Literal, Tuple
from copy import deepcopy


Device = Literal['cuda', 'mps', 'cpu']
_TorchInput = Tuple[Tensor, Any, Any]
_TorchOutput = Tensor

class _BaseHook:
    '''Base Hook is designed to support common features among all other hooks definition after inheritance. These include mostly debugging features that allow for some computation testing'''
    def __init__(self, debug=False, dynamic_memory=False) -> None:
        self._debug_mode: bool = debug
        self._dynamic_memory: bool = dynamic_memory
    ##

    @property
    def debug_mode(self):
        pass
    ##
    @debug_mode.getter
    def debug_mode(self):
        return self._debug_mode
    ##
    @debug_mode.setter
    def debug_mode(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError('cannot assign non-boolean vector to debug_mode attribute')
        self._debug_mode = value
    ##

    @property
    def dynamic_memory(self):
        pass
    ##
    @dynamic_memory.getter
    def dynamic_memory(self):
        return self._dynamic_memory
    ##
    @dynamic_memory.setter
    def dynamic_memory(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError('cannot assign non-boolean vector to dynamic_memory attribute')
        self._dynamic_memory = True
    ##

    def _debug(self):
        '''Sets internal state to "debug", causes the hook to print messages during inference tests'''
        self.debug_mode = True
    def _run(self):
        '''Sets internal state to "run", avoiding additional debugging messages in both forward and backwards call'''
        self.debug_mode = False
    ##

    def _dynamic_allocation(self):
        '''Changes internal state in order to dynamically load layers to GPU accelerator.'''
        self.dynamic_memory = True
    def _static_allocation(self):
        '''Changes internal state in order to handle allocation statically'''
        self.dynamic_memory = False
    ##

    def _device(self) -> Device:
        '''Returns the main GPU device used to accelerate computation, or 'cpu' is no GPU is found. Relies on `torch` detection methods'''
        if cuda.is_available():
            return 'cuda'
        elif mps.is_available():
            return 'mps'
        else:
            return 'cpu'  
    ##

    def pre_forward_hook(
            self,
            module: Module,
            input: _TorchInput,
    ):
        if self.debug_mode:
            print()
            print(f'[{module._get_name()}, pre-for]: received input {type(input)}')
            print(f'[{module._get_name()}, pre-for]: input shape: {input[0].shape}')
        if self.dynamic_memory:
            if self.debug_mode:
                print(f'[{module._get_name()}, pre-for]: moving layer to {self._device()}')
            module = module.to(self._device())
            mps.synchronize()
            mps.empty_cache()
    ##

    def forward_hook(
            self,
            module: Module,
            input: _TorchInput,
            output: Tensor
    ):
        if self.debug_mode:
            print(f'[{module._get_name()}, for]: resulted output: {type(output)}')
            print(f'[{module._get_name()}, for]: output shape: {output[0].shape}')
        if self._dynamic_memory:
            if self.debug_mode:
                print(f'[{module._get_name()}, for]: moving layer back to cpu')
            mps.synchronize()
            mps.empty_cache()
            module = module.to('cpu')
    ##

    # TODO: complete with these implementations, might be useful for fine-tuning
    def backward_hook(self):
        pass
##

class SteeringHook(_BaseHook):

    def __init__(self, injection_activation: Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert(isinstance(injection_activation, Tensor)), 'expected tensor as injector'
        # further assertion on shape of the tensor?
        self._inject = injection_activation
        if self.debug_mode:
            print(f'Assigned injection hook of shape {self._inject.shape}')
    ##


    def _test_if_input_changed(self,t1,t2,module) -> bool:
        '''Needed for debug in `pre_forward_hook()` call
        
        t1: steered input, expected size=[B, TOKENS_NUM, EMBEDS_DIM]\n
        t2: non-steered input, expected size=[B, TOKENS_NUM, EMBEDS_DIM]
        '''
        
        print(f'[{module._get_name()}, _test_if_input_changed]: testing equivalence with \'tol=1e-4\', consider editing this value for any issue')
        return all([allclose(dim2, self._inject, atol=1e-4) for dim1 in t1-t2 for dim2 in dim1])
    ##

    def pre_forward_hook(self, module, input: _TorchInput):
        '''Injects activations into residual stream'''
        super().pre_forward_hook(module, input)
        if self.debug_mode:
            print(f'[{module._get_name()}, pre-for]: injecting CAA vector')
            input_copy = deepcopy(input)
        
        # synchronize devices and cache orginal injection position
        inject_device_cache = self._inject.device
        self._inject = self._inject.to(input[0].device)
        edited_input = (input[0] + self._inject, *input[1:])
        self._inject.to(inject_device_cache)
        
        # eventual debug
        if self.debug_mode:
            if self._test_if_input_changed(edited_input[0], input_copy[0], module):
                pass
            else:
                RuntimeError(f'[{module._get_name()}, pre-for]: injection failed')

        return edited_input
    ##
##
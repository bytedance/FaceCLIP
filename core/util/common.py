# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import functools
import importlib
import re
import logging
import torch
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

def get_module_by_name(module, name):
    names = name.split(sep='.')
    return functools.reduce(getattr, names, module)

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params

def instantiate_from_config(config: OmegaConf, **kwargs):
    target = OmegaConf.structured(config)["target"]
    if target is None:
        msg = "Expected key `target` to instantiate."
        raise KeyError(msg)
    config_params = OmegaConf.structured(config)["params"]
    if config_params is None:
        config_params = {}
    log.info("Instantiating %s: %s", target, config_params)

    klass = get_obj_from_str(target)
    params = config_params

    if hasattr(config, 'from_pretrained'):
        model_path = config.from_pretrained
        config_dict = dict(config)
        
        if 'precision' in config_dict:
            dtype = get_obj_from_str(config_dict.pop('precision'))
        else:
            dtype = torch.float32
        
        if hasattr(config, 'sub_folder'):
            return klass.from_pretrained(
                model_path,
                use_safetensors=True,
                subfolder=config.sub_folder,
                precision=dtype,
                **config_dict
            ).to(dtype)
        else:
            return klass.from_pretrained(
                model_path,
                use_safetensors=True,
                precision=dtype,
                **config_dict
            ).to(dtype)
        
    try:
        return klass(**params, **kwargs)
    except:
         return klass(config=params)

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def customfn_formatter(customfn_str: str, ensure_dict: bool = True):
    assert customfn_str[-1] == ")" and customfn_str.startswith(
        "functools.partial("
    ), "Custom funcStage should be in the format: 'functools.partial(custom_func, {params})'"
    split = customfn_str.replace("functools.partial(", "")[:-1].split(",")
    if ensure_dict:
        try:
            prms = eval(",".join(split[1:]))
        except (SyntaxError, NameError, TypeError, ValueError):
            assert (
                False
            ), f"The parameters - {','.join(split[1:])} - could not be evaluated."
        assert isinstance(
            prms, dict
        ), "The parameters should be in the form of a dictionary"
    else:
        parameters = ",".join(split[1:])
        parameters = re.sub(r"[\[\]\ ]", "", parameters).split(",")
        prms = [eval(x) if x.isnumeric() else x for x in parameters]
    return split[0], prms

def get_callable_from_str(string: str):
    if "functools.partial" in string:
        fn_str, prms = customfn_formatter(string, ensure_dict=False)
        return functools.partial(get_obj_from_str(fn_str), prms)
    else:
        return get_obj_from_str(string)

def make_model_fixed(model):
    def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self

    model = model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad = False
    return model


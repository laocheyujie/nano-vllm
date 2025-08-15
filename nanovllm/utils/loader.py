import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    # 把 loaded_weight 的值拷贝到 param.data
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        # param 是 torch.nn.Parameter 实例，可以用 param.data 访问参数矩阵
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        # weight_loader 在 QKVParallelLinear 中被重写，增加了 shard_id 参数
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 根据 weight_name 获取对应的参数属性
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))

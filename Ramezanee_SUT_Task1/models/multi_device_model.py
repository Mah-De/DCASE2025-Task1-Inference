import torch
import torch.nn as nn
import copy
from typing import Tuple

from Ramezanee_SUT_Task1.models.net import RepConv2d, ReParametrize
import importlib.resources as pkg_resources
from importlib.resources import as_file, files
from Ramezanee_SUT_Task1 import ckpts

# def ReParametrize(module,device):
#     """
#     Recursively replaces all RepConv2d layers in a module with their reparametrized version.
#     """
#     for name, child in list(module.named_children()):
#         if isinstance(child, RepConv2d):
#             # print(f"Reparametrizing {name}")
#             new_module = child.get_reparametrized_layer().to(device)
#             setattr(module, name, new_module)
#         else:
#             ReParametrize(child,device)


class MultiDeviceModelContainer(nn.Module):
    """
    Multiple device fine-tuning container.
    """
    def __init__(self, base_model: nn.Module, devices: list, submission = 4):
        """
        Initializes the container with a base model.

        Args:
            base_model (nn.Module): The base model to be adapted per device.
            devices (list): A list of device identifiers (e.g., ["a", "b", "c"]).
        """
        super().__init__()
        self.base_model = base_model
        self.devices = devices

        # Define the base resource directory
        ckpts_dir = files(ckpts)

        # Use `as_file` to ensure we get a real temporary path (if needed)
        with as_file(ckpts_dir.joinpath(f"V{submission}-A.pth")) as a_path, \
            as_file(ckpts_dir.joinpath(f"V{submission}-B.pth")) as b_path, \
            as_file(ckpts_dir.joinpath(f"V{submission}-C.pth")) as c_path, \
            as_file(ckpts_dir.joinpath(f"V{submission}-S1.pth")) as s1_path, \
            as_file(ckpts_dir.joinpath(f"V{submission}-S2.pth")) as s2_path, \
            as_file(ckpts_dir.joinpath(f"V{submission}-S3.pth")) as s3_path, \
            as_file(ckpts_dir.joinpath("GLOBAL.pth")) as global_path:

            model_names = {
                "unknown": str(global_path),
                "a": str(a_path),
                "b": str(b_path),
                "c": str(c_path),
                "s1": str(s1_path),
                "s2": str(s2_path),
                "s3": str(s3_path),
            }


        # Create device-specific models
        self.device_models = nn.ModuleDict({})
        for dev in devices : 
            self.device_models[dev] = copy.deepcopy(base_model)
            self.device_models[dev].load_state_dict(torch.load(model_names[dev], map_location='cpu')['model_state_dict'])

    def forward(self, x: torch.Tensor, devices: Tuple[str] = None) -> torch.Tensor:
        """
        Forward pass through the model specific to the given device.

        Args:
            x (torch.Tensor): Input tensor.
            devices (Tuple[str]): Tuple of device identifiers corresponding to each sample.

        Returns:
            torch.Tensor: The model output.
        """
        if devices is None:
            # No device info → use base model
            return self.base_model(x)
        elif len(set(devices)) > 1:
            # More than one device in batch → forward one sample at a time
            return self._forward_multi_device(x, devices)
        elif devices[0] in self.device_models:
            # Single known device → use device-specific model
            return self.get_model_for_device(devices[0])(x)
        else:
            # Single unknown device → fall back to base model
            return self.base_model(x)

    def _forward_multi_device(self, x: torch.Tensor, devices: Tuple[str]) -> torch.Tensor:
        """
        Handles forward pass when multiple devices are present in the batch.
        """
        outputs = [self.device_models[device](x[i].unsqueeze(0)) if device in self.device_models
                   else self.base_model(x[i].unsqueeze(0))
                   for i, device in enumerate(devices)]
        return torch.cat(outputs)

    def get_model_for_device(self, device_name: str) -> nn.Module:
        """
        Retrieve the model corresponding to a specific device.

        Args:
            device_name (str): The device identifier.

        Returns:
            nn.Module: The model corresponding to the device.
        """
        if device_name in self.device_models:
            return self.device_models[device_name]
                
        else:
            return self.base_model
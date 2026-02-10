# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained DiT models
"""
from torchvision.datasets.utils import download_url
import torch
import os


pretrained_models = {'DiT-XL-2-512x512.pt', 'DiT-XL-2-256x256.pt'}

# Cache directory for downloaded models
CACHE_DIR = '/scratch/daniela/.cache/finediffusion'


def find_model(model_name):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        return download_model(model_name)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        return checkpoint

# def find_model(model_name):
#     """
#     Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
#     """
#     if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
#         return download_model(model_name)
#     else:  # Load a custom DiT checkpoint:
#         assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
#         checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
#         # Return full checkpoint dict if it's a training checkpoint (has 'model' key)
#         # Only extract 'ema' if it's a pretrained model without a 'model' key
#         if "model" not in checkpoint and "ema" in checkpoint:
#             checkpoint = checkpoint["ema"]
#         return checkpoint

def download_model(model_name):
    """
    Downloads a pre-trained DiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = os.path.join(CACHE_DIR, model_name)
    if not os.path.isfile(local_path):
        os.makedirs(CACHE_DIR, exist_ok=True)
        web_path = f'https://dl.fbaipublicfiles.com/DiT/models/{model_name}'
        download_url(web_path, CACHE_DIR)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


if __name__ == "__main__":
    # Download all DiT checkpoints
    for model in pretrained_models:
        download_model(model)
    print('Done.')

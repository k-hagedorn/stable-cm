import torch as th
import os

def find_model(model_name):
    """
    Finds a pre-trained SiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find SiT checkpoint at {model_name}'
    checkpoint = th.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint
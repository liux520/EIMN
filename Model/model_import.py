import torch
import os


def model_import(model_name, scale, load_weights=False):
    root = os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)), 'Weights')

    if model_name == 'EIMN_L':
        from Model.EIMN import EIMN_L
        model = EIMN_L(scale=scale)

    elif model_name == 'EIMN_A':
        from Model.EIMN import EIMN_A
        model = EIMN_A(scale=scale)

    if load_weights:
        weight_path = os.path.join(root, f'{model_name}_x{scale}.pth')
        checkpoint = torch.load(weight_path, map_location='cpu')
        print(f'{model_name}_x{scale}_ckpt')
        model.load_state_dict(checkpoint)

    return model


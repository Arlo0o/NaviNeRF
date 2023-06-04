import json
import numpy as np
import torch
from torch import nn
import sys
 
from models.gan_with_shift import gan_with_shift
sys.path.append("..") 
from nerfgan_generate import nerfgan
 


@gan_with_shift
def make_nerfgan(network_pkl = 'ffhq_256.pkl'):
    model = nerfgan( skip=False, render_program=None, network_pkl = network_pkl).cuda()
    setattr(model, 'dim_z', [512])
    return model


@gan_with_shift
def make_nerfgan_skip_mapping(network_pkl2 = 'ffhq_256_v2.pkl'):
    model = nerfgan( skip=True, render_program=None, network_pkl2 = network_pkl2 ).cuda()
    setattr(model, 'dim_z', [512])
    return model




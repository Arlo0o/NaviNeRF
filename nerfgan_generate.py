import os
import re
import time
import glob
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import imageio
import legacy
from renderer import Renderer
import argparse
import os
 
###----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------
os.environ['PYOPENGL_PLATFORM'] = 'egl'

parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--network', default=None)
parser.add_argument('--seeds', default=[0], type=num_range, help='List of random seeds')
parser.add_argument('--trunc',  default=0.7, type=float, help='Truncation psi')
parser.add_argument('--class_idx', type=int, help='Class label (unconditional if not specified)')
parser.add_argument('--noisemode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const')
parser.add_argument('--projectedw', help='Projection result file', type=str, metavar='FILE')
parser.add_argument('--outdir', default = "./", help='Where to save the output images', type=str,  metavar='DIR')
parser.add_argument('--render-program', default="rotation_camera")
parser.add_argument('--render-option', default=None, type=str, help="e.g. up_256, camera, depth")
parser.add_argument('--n_steps', default=8, type=int, help="number of steps for each seed")
parser.add_argument('--no-video', default=False)
parser.add_argument('--relative_range_u_scale', default=1.0, type=float, help="relative scale on top of the original range u")
args = parser.parse_args()


import torch.nn as nn
class nerfgan(nn.Module):
    def __init__(self,          
        network_pkl=args.network,
        seeds=[ 4],
        truncation_psi=args.trunc,
        noise_mode=args.noisemode,
        outdir=args.outdir,
        class_idx=args.class_idx,
        projected_w=args.projectedw,
        render_program=None ,
        render_option=None,
        n_steps=8,
        no_video=False,
        relative_range_u_scale=2.0,
        skip=True
        ):


        
        super(nerfgan, self).__init__()
        self.truncation_psi=truncation_psi
        self.noise_mode=noise_mode
        self.outdir=outdir
        self.class_idx=class_idx
        self.projected_w=projected_w
        self.render_program=render_program
        self.n_steps=n_steps
        self.no_video=no_video
        self.relative_range_u_scale=relative_range_u_scale
        self.render_option=render_option
        self.seeds=seeds
        self.network_pkl=network_pkl
        self.skip=skip


    def forward(self, parameter_z, z_initial= None):
    
        def stack_imgs(imgs):
            img = torch.stack(imgs, dim=2)
            return img.reshape(img.size(0) * img.size(1), img.size(2) * img.size(3), 3)

        def proc_img(img):
            return (  img.permute(0, 2, 3, 1) * 127.5 + 128  ).clamp(0, 255).to(torch.uint8).cpu()
        network_pkl=self.network_pkl
        truncation_psi=self.truncation_psi
        noise_mode=self.noise_mode
        outdir=self.outdir
        class_idx=self.class_idx
        projected_w=self.projected_w
        render_program=self.render_program
        n_steps=self.n_steps
        no_video=self.no_video
        relative_range_u_scale=self.relative_range_u_scale
        render_option=self.render_option
        seeds=self.seeds
        skip = self.skip
        device = torch.device('cuda')


        if os.path.isdir(network_pkl):
            network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]

        with dnnlib.util.open_url(network_pkl) as f:
            network = legacy.load_network_pkl(f)
            G = network['G_ema'].to(device) # type: ignore
            D = network['D'].to(device)


        # Labels.
        label = torch.zeros([1, G.c_dim], device=device)
        if G.c_dim != 0:
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print ('warn: --class=lbl ignored when running on an unconditional network')


        from training.networks import Generator
        from torch_utils import misc
        with torch.no_grad():
            G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
            misc.copy_params_and_buffers(G, G2, require_all=False)
        G2 = Renderer(G2, D, program=render_program,skip=skip)


        all_imgs = []

        if projected_w is not None:


            ws = projected_w
            img = G2(styles=ws, truncation_psi=truncation_psi, noise_mode=noise_mode, render_option=render_option, skip=skip)
            out = img
            if isinstance(img, List):
                imgs = [proc_img(i) for i in img]
                all_imgs += [imgs]
            else:
                img = proc_img(img)[0]

        else:
            for seed_idx, seed in enumerate(seeds):
      
  
                G2.set_random_seed(seed) 

           

                relative_range_u = [0.5 - 0.5 * relative_range_u_scale, 0.5 + 0.5 * relative_range_u_scale]
                outputs = G2(
                    z=parameter_z,
                    z_initial= z_initial,
                    c=label,
                    truncation_psi=truncation_psi,
                    noise_mode=noise_mode,
                    render_option=render_option,
                    n_steps=n_steps,
                    relative_range_u=relative_range_u,
                    return_cameras=True,
                    skip=skip)
        
                if isinstance(outputs, tuple):
                    img, cameras = outputs
                else:
                    img = outputs
                   
                out = img
                return out

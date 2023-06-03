import os
import argparse
import random
import torch
import torch.multiprocessing.reductions

import matplotlib
matplotlib.use("Agg")
 


from constants import DEFORMATOR_TYPE_DICT, SHIFT_DISTRIDUTION_DICT, WEIGHTS
from loading import load_generator
from latent_deformator import LatentDeformator
from latent_shift_predictor import LatentShiftPredictor, LeNetShiftPredictor
from trainer import Trainer, Params
from visualization import inspect_all_directions
from utils import make_noise, save_command_run_params


def main():
    parser = argparse.ArgumentParser(description='Latent space rectification')
    for key, val in Params().__dict__.items():
        target_type = type(val) if val is not None else int
        parser.add_argument('--{}'.format(key), type=target_type, default=None)

    parser.add_argument('--out', type=str, default = "output", help='results directory')
    parser.add_argument('--gan_type', default = "nerfgan" , type=str, choices=WEIGHTS.keys(), help='_skip_mapping, nerfgan')
    parser.add_argument('--deformator_model_path', type=str, default= None)
    parser.add_argument('--shift_predictor_model_path', type=str, default= None) 

    parser.add_argument('--deformator', type=str, default='ortho',
                        choices=DEFORMATOR_TYPE_DICT.keys(), help='deformator type')
    parser.add_argument('--deformator_random_init', type=bool, default=True)
    parser.add_argument('--load_skip', type=bool, default=False)
    parser.add_argument('--shift_predictor_size', type=int, help='reconstructor resolution')
    parser.add_argument('--shift_predictor', type=str,
                        choices=['ResNet', 'LeNet'], default='ResNet', help='reconstructor type')
    parser.add_argument('--shift_distribution_key', type=str,
                        choices=SHIFT_DISTRIDUTION_DICT.keys())

    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--device', type=int, default=0 )
    parser.add_argument('--multi_gpu', type=bool, default=False,
                        help='Run generator in parallel. Be aware of old pytorch versions:\
                              https://github.com/pytorch/pytorch/issues/17345')
    # model-specific
    parser.add_argument('--target_class', nargs='+', type=int, default=[239],
                        help='classes to use for conditional GANs')
    parser.add_argument('--w_shift', type=bool, default=False,
                        help='latent directions search in w-space for StyleGAN2')
    parser.add_argument('--gan_resolution', type=int, default=1024,
                        help='generator out images resolution. Required only for StyleGAN2')

    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    save_command_run_params(args)

 
    weights_path = WEIGHTS[args.gan_type]


    G = load_generator(args.__dict__, weights_path)

    deformator = LatentDeformator(shift_dim=G.dim_shift,
                                  input_dim=args.directions_count,
                                  out_dim=args.max_latent_dim,
                                  type=DEFORMATOR_TYPE_DICT[args.deformator],
                                  random_init=args.deformator_random_init).cuda()
    

    if args.shift_predictor == 'ResNet':
        shift_predictor = LatentShiftPredictor(
            deformator.input_dim, args.shift_predictor_size).cuda()
    elif args.shift_predictor == 'LeNet':
        shift_predictor = LeNetShiftPredictor(
            deformator.input_dim, 1 if args.gan_type == 'SN_MNIST' else 3).cuda()

    # training
    args.shift_distribution = SHIFT_DISTRIDUTION_DICT[args.shift_distribution_key]

    params = Params(**args.__dict__)
    params.directions_count = int(deformator.input_dim)
    params.max_latent_dim = int(deformator.out_dim)


    ####------------------------------------------------LOADING-------------------------------------------------------------------###
    skipping_G = None
    
    if args.gan_type=='nerfgan' and args.load_skip:
        print("loading pretrained deformator model...")  
        args.deformator='linear'  ### A---> linear

 
        deformator.load_state_dict( torch.load(args.deformator_model_path, map_location=torch.device('cpu')))
        shift_predictor.load_state_dict(  torch.load(args.shift_predictor_model_path, map_location=torch.device('cpu')))

        from models.gan_load import make_nerfgan_skip_mapping
        skipping_G = make_nerfgan_skip_mapping()
 
    trainer = Trainer(params, out_dir=args.out)
    trainer.train(G,  deformator, shift_predictor, multi_gpu=args.multi_gpu, skipping_G=skipping_G)
    

    save_results_charts_withmapping(G, deformator, params, trainer.log_dir)


def save_results_charts_skipmapping(G, deformator, params, out_dir):
    deformator.eval()
    G.eval()
    z = make_noise(3, G.dim_z, params.truncation).cuda()
    
    inspect_all_directions(
        G, deformator, os.path.join(out_dir, 'skipmapping_charts_s{}'.format(int(params.shift_scale))),
        zs=z, shifts_r=params.shift_scale)
    inspect_all_directions(
        G, deformator, os.path.join(out_dir, 'skipmapping_charts_s{}'.format(int(3 * params.shift_scale))),
        zs=z, shifts_r=3 * params.shift_scale)


def save_results_charts_withmapping(G, deformator, params, out_dir):
    deformator.eval()
    G.eval()
    z = make_noise(3, G.dim_z, params.truncation).cuda()
    inspect_all_directions(
        G, deformator, os.path.join(out_dir, 'withmapping_charts_s{}'.format(int(params.shift_scale))),
        zs=z, shifts_r=params.shift_scale)
    inspect_all_directions(
        G, deformator, os.path.join(out_dir, 'withmapping_charts_s{}'.format(int(3 * params.shift_scale))),
        zs=z, shifts_r=3 * params.shift_scale)



if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print("all time: ", (time.time()-start)/3600 )

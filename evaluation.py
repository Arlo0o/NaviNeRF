import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from torch_tools.visualization import to_image
from visualization import interpolate
from loading import load_from_dir
import numpy as np
import os

deformator, G, shift_predictor = load_from_dir(
    './output',
    G_weights=None,
    )



from utils import is_conditional
rows = 30
plt.figure(figsize=(5, rows), dpi=250)
if is_conditional(G):
    G.set_classes(12)


zs = torch.from_numpy(np.random.RandomState(1).uniform(-1,1,(rows, 512))).cuda()
 

for j in range(0,512):
        for z, i in zip(zs, range(rows)):

                interpolation_deformed = interpolate(
                    G, z.unsqueeze(0),
                    shifts_r=2.5,  ## 2.5
                    shifts_count=2.5 *2/(6.1),  ### 2.5
                    dim=int(j),
                    deformator=deformator,
                    with_central_border=True, index0 = j, index=i)
                plt.subplot(rows, 1, i + 1)
                plt.axis('off')
                grid = make_grid(interpolation_deformed, nrow=11, padding=1, pad_value=0.0)
                grid = torch.clamp(grid, -1, 1)
                print("processing...", j, "_", i)
                plt.imshow(to_image(grid))
                plt.savefig("./out%d.png" %(j) )
  

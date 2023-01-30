import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch

def visualize_data(dataloader, num_images, figsize = (10,10)):
    '''
    Plot the images

    Args:
    - dataloader (Pytorch Dataloader): Pytorch Dataloader
    - num_images (int): Number of images to be plotted (Plots the nearest perfect square nummber of images)
    - figsize (tuple): figure size
    '''
    # Inverse Transform for imagenet stats
    inv_transforms = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                      transforms.Normalize(mean = [-0.485, -0.456, -0.406],
                                         std = [ 1., 1., 1. ])])
    grid_size = int(np.sqrt(num_images))
    fig, axes = plt.subplots(nrows = grid_size, ncols = grid_size, figsize = figsize)
    total_imgs, current_img = grid_size*grid_size, 0
    for imgs, ages, genders in dataloader:
        for idx, img in enumerate(imgs):
            age = ages[idx]
            gender = genders[idx]
            img = np.transpose(inv_transforms(img).detach().cpu().numpy(), (1,2,0))
            axes[current_img//grid_size, current_img%grid_size].imshow(img)
            axes[current_img//grid_size, current_img%grid_size].axis('off')
            axes[current_img//grid_size, current_img%grid_size].set_title(f"Age: {age}, Gender: {gender}")
            current_img+=1
            if current_img == total_imgs:
                plt.show()
                return
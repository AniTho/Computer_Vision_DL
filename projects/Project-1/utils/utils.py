import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch
import cv2

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
            axes[current_img//grid_size, current_img%grid_size].set_title(f"Age: {denormalize_age(age)}, Gender: {get_gender(gender)}")
            current_img+=1
            if current_img == total_imgs:
                plt.show()
                return

def build_model(model, epochs, lr = 1e-04, schedule = False):
    '''
    Build the loss, optimizer and scheduler used for training model

    Args:
    - model (torch.nn): Model object used
    - epochs (int): Number of maximum iteration
    - lr (float): Learning rate for optimizer
    - schedule (bool): To use learning rate scheduler or not

    Returns:
    - (torch.nn): Mean squared error loss fxn
    - (torch.nn): Binary cross entropy loss fxn
    - (torch.optim): Adam optimizer
    - (torch.optim): [Optional] Learning Rate Scheduler
    '''
    criterion_mse = torch.nn.L1Loss()
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-07)
        return criterion_mse, criterion_bce, optimizer, scheduler
    return criterion_mse, criterion_bce, optimizer
    
def save_checkpoint(model, optimizer = None, file_path = 'checkpoint.pt'):
    '''
    Save checkpoints for loading later and resuming training

    Args:
    - model (torch.nn): Neural network model to be saved
    - optimizer (torch.optim): Current state of optimizer
    - file_path (str): Path to save the checkpoint
    '''
    save_file = {'model_state_dict': model.state_dict()}
    if optimizer:
        save_file['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(save_file, file_path)
    
def load_checkpoint(model, optimizer = None, file_path = 'checkpoint.pt'):
    '''
    Load checkpoints

    Args:
    - model (torch.nn): Neural network model to be loaded
    - optimizer (torch.optim): Optimizer data to be loaded
    - file_path (str): Path where the checkpoint is saved

    Returns:
    - (torch.nn): Model with weights loaded from checkpoint
    - (torch.optim): Optimizer state loaded from checkpoint
    '''
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and ('optimizer_state_dict' in checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def denormalize_age(age):
    return int(age*(115.0 - 1.0) + 1.0)

def get_gender(gender):
    return 'female' if gender == 0 else 'male'

def plot_losses(train_loss, valid_loss, title = "Loss vs epochs", figsize = (4,4)):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(train_loss, 'r--', label = 'Train')
    plt.plot(valid_loss, 'g-', label = 'Valid')
    plt.legend()
    plt.show()

def calculate_accuracy(dataloader, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        model.eval()
        total = 0
        num_correct = 0
        for imgs, _, genders in dataloader:
            imgs = imgs.to(device, non_blocking = True)
            genders = genders.to(device, non_blocking = True).float()
            _, pred_genders = model(imgs)
            pred_genders = (pred_genders >= 0.5).float().squeeze()
            num_correct += (pred_genders == genders).sum().item()
            total += len(genders)
        
        print(f"Final Accuracy: {num_correct/total*100:.3f}")

def gradcam_visualization(img, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inv_transforms = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                      transforms.Normalize(mean = [-0.485, -0.456, -0.406],
                                         std = [ 1., 1., 1. ])])
    img = img.to(device)[None]
    model = model.to(device)
    activation_network = model.feature_extractor[:8]
    age, gender = model(img)
    feature_maps = activation_network(img)
    model.zero_grad()
    age.backward(retain_graph = True)
    gender.backward(retain_graph = True)
    gradients_mean = model.feature_extractor[7][-1].conv3.weight.grad.data.mean((1,2,3))
    for i in range(len(gradients_mean)):
        feature_maps[:,i, :, :] *= gradients_mean[i]
    heatmap = torch.mean(feature_maps, dim = 1)[0].cpu().detach()
    heatmap = heatmap.numpy()
    min_value, max_value = heatmap.min(), heatmap.max()
    heatmap = 255*((heatmap - min_value)/(max_value - min_value))
    heatmap = heatmap.astype(np.uint8)
    img = 255*np.transpose(inv_transforms(img[0]).detach().cpu().numpy(), (1,2,0))
    heatmap = cv2.resize(heatmap, img.shape[:-1])
    heatmap = 255 - heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET).astype(np.uint8)
    heatmap = (0.3*heatmap + 0.7*img).astype(np.uint8)
    plt.imshow(heatmap)
    plt.show()
import split_data
from dataload.dataload import load_data
from utils.utils import load_checkpoint, build_model
from model.resnet50 import create_model
import torch
from train import train

def main():
    images_dir = 'data/UTKFace/images'
    base_data_dir = 'data/UTKFace'
    train_split = 0.7
    valid_split = 0.2

    # Hyperparameters
    aug_p = 0.4
    batch_size = 64
    base_model = 'resnet50'
    dropout_p = 0.4
    freeze = True
    num_layers = 120
    learning_rate = 1e-03
    unfrozen_learning_rate = 3e-05
    epochs = 20
    split_data.split(base_data_dir, images_dir, train_split, valid_split)
    trainloader, validloader, testloader = load_data(base_data_dir, images_dir, aug_p, batch_size)
    model = create_model(base_model, freeze, num_layers, dropout_p)
    criterion_mae, criterion_bce, optimizer, scheduler = build_model(model, epochs, learning_rate, True)
    model, train_bce_losses, train_mae_losses, valid_bce_losses, valid_mae_losses, \
    best_loss = train(trainloader, validloader, model, criterion_mae, criterion_bce, optimizer, epochs, scheduler = scheduler)

    # Unfreeze
    model = create_model(base_model, freeze = False, dropout_p=dropout_p)
    model = load_checkpoint(model, file_path='saved_models/multi_task.pt')
    criterion_mae, criterion_bce, optimizer, scheduler = build_model(model, epochs, unfrozen_learning_rate, True)
    model, train_bce_losses, train_mae_losses, valid_bce_losses, valid_mae_losses, \
    best_loss = train(trainloader, validloader, model, criterion_mae, criterion_bce, optimizer, epochs, best_loss, scheduler)

    model_scripted = torch.jit.script(model)
    model_scripted.save('saved_models/multitask_final.pt')


if __name__ == "__main__":
    main()
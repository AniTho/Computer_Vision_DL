import split_data
from dataload.dataload import load_data
from utils.utils import visualize_data

def main():
    images_dir = 'data/UTKFace/images'
    base_data_dir = 'data/UTKFace'
    train_split = 0.7
    valid_split = 0.2

    # Hyperparameters
    aug_p = 0.4
    batch_size = 64
    split_data.split(base_data_dir, images_dir, train_split, valid_split)
    trainloader, validloader, testloader = load_data(base_data_dir, images_dir, aug_p, batch_size)

if __name__ == "__main__":
    main()
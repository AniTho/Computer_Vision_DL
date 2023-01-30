import split_data
from dataload.dataload import load_data

def main():
    images_dir = 'data/UTKFace/images'
    train_split = 0.7
    valid_split = 0.2
    split_data.split(images_dir, train_split, valid_split)
    trainloader, validloader, testloader = load_data()

if __name__ == "__main__":
    main()
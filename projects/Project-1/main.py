import split_data
import os

def main():
    images_dir = 'data/UTKFace/images'
    train_split = 0.8
    split_data.split(images_dir, train_split)

if __name__ == "__main__":
    main()
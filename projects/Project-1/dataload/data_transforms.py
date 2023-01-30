import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(aug_p):
    '''
    Returns the augmentations for train, valid and test set

    Args:
    - aug_p (float): Ranging between [0, 1], denotes the probability of applying transformation

    Returns:
    - (albumentation compose object): Object containing the transforms for train dataset
    - (albumentation compose object): Object containing the transforms for valid dataset
    - (albumentation compose object): Object containing the transforms for test dataset
    '''
    resize_side = 256
    output_image_crop = (224, 224)

    # Stats taken from imagenet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transforms = A.Compose([A.LongestMaxSize(max_size=resize_side, interpolation=1),
                                 A.PadIfNeeded(min_height=resize_side, min_width=resize_side, border_mode=0, value=(0,0,0)),
                                 A.ShiftScaleRotate(shift_limit=(0.1,0.1), scale_limit=0.05, rotate_limit=15, p=aug_p),
                                 A.RandomCrop(height = output_image_crop[0], width = output_image_crop[1]),
                                 A.HorizontalFlip(p=aug_p),
                                 A.Normalize(mean, std, max_pixel_value=1.0),
                                 ToTensorV2(),
                                 ])
    valid_transforms = A.Compose([A.LongestMaxSize(max_size=resize_side, interpolation=1),
                                 A.PadIfNeeded(min_height=resize_side, min_width=resize_side, border_mode=0, value=(0,0,0)),
                                 A.CenterCrop(height = output_image_crop[0], width = output_image_crop[1]),
                                 A.Normalize(mean, std, max_pixel_value=1.0),
                                 ToTensorV2(),
                                 ])
    test_transforms = A.Compose([A.LongestMaxSize(max_size=resize_side, interpolation=1),
                                 A.PadIfNeeded(min_height=resize_side, min_width=resize_side, border_mode=0, value=(0,0,0)),
                                 A.CenterCrop(height = output_image_crop[0], width = output_image_crop[1]),
                                 A.Normalize(mean, std, max_pixel_value=1.0),
                                 ToTensorV2(),
                                 ])
    return train_transforms, valid_transforms, test_transforms
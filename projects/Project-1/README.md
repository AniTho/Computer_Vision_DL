# Age and Gender prediction

The projects aims to develop a system to detect the age and gender of the person from their image.

## Checklist:
- [x] Split the dataset and create csv so as to not change train, valid and test dataset at each run
- [x] Create Dataloader
- [ ] Build a model for age and gender detection from images
- [ ] Detect human from the image and pass each detected person through the model for detection
- [ ] Finally deploy it to detect persons from video, draw a bounding boxes around them to try and predict the age and gender of the detected person.

## Dataset Citation:
@inproceedings{zhifei2017cvpr,
  title={Age Progression/Regression by Conditional Adversarial Autoencoder},
  author={Zhang, Zhifei, Song, Yang, and Qi, Hairong},
  booktitle={IEEE Conference on Computer Visiogn and Pattern Recognition (CVPR)},
  year={2017},
  organization={IEEE}
}

__NOTE:__ Image with name 53__0_20170116184028385.jpg is removed from dataset due to not following the pattern for label.
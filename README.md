# Computer_Vision_DL
Learning Computer Vision Techniques using deep learning in 2023

__Day 001 Progress:__ Learnt about Artificial Neural Network(ANN) and built a simple ANN using pytorch and trained it on Fashion MNIST Dataset

__Day 002 Progress:__ Learnt about hyperaparameter tuning in ANN and tried some of those approaches. Some of the approach is left to take care of overfitting.

__Day 003 Progress:__ Learnt about how to take care of overfitting via techniques such as dropout, L1 and L2 regularization. Also made the network a bit deeper without overfitting to the data. Finally put all the hyperparameters and techniques tried in Day_02 notebook and created final model.

__Day 004 Progress:__ Learnt about Convolutional Neural Network (CNN) and its components. Also updated Day_03 notebook to save model weights. In Day_04 notebooks ran some experiment why ANN is not useful for images and implemented a very basic convolutional operation block and pooling block.

__Day 005 Progress:__ Studied LeNet architecture and implemented it using pytorch (with some modification) and trained it on Fashion MNIST dataset. Also learnt about data augmentation and started to implement some of the functions and their visualization using kornia library.

__Day 006 Progress:__ Trained final convolution neural network (inspired from LeNet) on fashion MNIST dataset with augmentation and shown the affect of translation on final trained model.

__Day 007 Progress:__ Optimized the data loader, since it is the part where most time is consumed and also emperically shown result of the comparison. Trained a VGG Net model using transfer learning. Also implemented mixed precision version of the training.

__Day 008 Progress:__ Used Resnet using timm library, also replaced Kornia augmentations with albumentation augmentations. Tried different weight initialization techniques.

__DAY 009 Progress:__ Started to work on the project to find age and gender of a person from image.

__DAY 010 Progress:__ Build the dataloader and added certain utility functions for visualization.

__DAY 011 Progress:__ Build the model and trained it.

__DAY 012 Progress:__ Fixed some bugs in the project. Also added a jupyter notebook for purposes of visualization. Read about visualizing what conv nets are learning and especially grad cams. Also added functionality to visualize the output and plotting losses. Also visually inspecting some data, the data is wrongly labelled which has to be corrected.

__DAY 013 Progress:__ Studied about gradcam in depth for visualizing what convenet is learning. Studied from several blogs and implemented in code.
Reference:

1. https://towardsdatascience.com/interpretability-in-deep-learning-with-w-b-cam-and-gradcam-45ba5296a58a
2. https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-grad-cam
3. https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82

__Day 014 Progress:__ Studied basics of object detection, building bounding boxes.

__Day 015 Progress:__ Played with felzenszwalb segmentation and read more about it and read about selective search as well and visualize it using library. Also studied about IoU. 

__Day 016 Progress:__ Finished IoU, studied about mAP and non max supression. Downloaded Coco dataset.

__Day 017 Progress:__ Explored data and fetch labels and bbox from coco json file and generated a csv from it

__Day 018 Progress:__ Written code for fetching region of interest from all the proposed region by selective search and the offset between true bounding box label and proposed one

__Day 019 Progress:__ Build Dataloader for RCNN and started building the architecture.

__Day 020 Progress:__ Trained RCNN from scratch on coco dataset.

__Day 021 Progress:__ Took a break from object detection algorithms and started studying image manipulation, starting with autoencoder. Also studied a bit about logging using wandb

__Day 022 Progress:__ Implemented vanilla encoder using fully connected network.

__Day 023 Progress:__ Tried some hyperparameter tuning and added visualization. Also tweaked implemented network.
 (Visualize Results: https://wandb.ai/aniketthomas/Autoencoder)

 __Day 024 Progress:__ Adding CNN based autoencoder and addin TSNE based visualization for different latent size
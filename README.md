
# Deep Eye Tracker with Tensorflow and Python

In this project, I took images of myself from a webcam and then performed data augmentation to make the final dataset of over nine thousand images.  Following this, I utilized TensorFlow's sequential API to construct a model specifically designed for Eye Tracking within these images.


## Dataset
* Collected images from webcam using OpenCV
* Annotated those images with LabelMe
* Divided the dataset into three subsets ( training, validation, and test).
* Performed augmentation using Albumentations

## Build  Deep Learning Model using the Sequential API
### Base Model:

* Utilized ResNet152V2 as the foundational model.
* Extracted without the fully connected top layers to leverage its feature extraction capabilities.
* Input is set to handle images of shape 250x250x3 (WxHxC).

### Additional Convolutional Layers:

* Enhanced the base model with subsequent convolutional layers to further refine and process image features.
* Layer sequences:
  * 512 filters, 3x3 kernel, ReLU activation.
  * 512 filters, 3x3 kernel, ReLU activation.
  * 256 filters, 3x3 kernel, stride of 2, ReLU activation.
  * 256 filters, 2x2 kernel, stride of 2, ReLU activation.
    
### Dropout Layer:

* Incorporated for regularization with a dropout rate of 0.05.
* Aims to reduce overfitting by intermittently dropping nodes during training.

### Final Convolution and Output Layer:

* The model culminates with a convolutional layer of:
  * 4 filters, 2x2 kernel, stride of 2.
* Subsequently, a reshape layer transforms the tensor to a shape of (4,).
* This flattened output gives the Keypoints for Two Eyes, where each eye has an (x, y) coordinate, thus resulting in four values.



This architecture is designed for eye tracking, precisely identifying the keypoints for each eye in the image.

## Define Losses and Optimizers
* Defined:
  * Optimizer
  * Learning rate
  * Loss

## Train Neural Network
* Trained the model over 10 epochs, allowing it to iteratively learn and optimize its performance on the detection and localization tasks.

## Making Predictions
* Predictions were made on the test set, evaluating the model's performance on unseen data.
* The model was further tested for real-time face detection using a webcam, demonstrating its capability to work with live video input.

## Credits
This project was inspired and guided by a wonderful tutorial by Nicholas Renotte. A big shout out to Nicholas for his clear and insightful instruction. You can check out the tutorial [here](https://www.youtube.com/watch?v=qpFrg4gN4Mg&t=4333s&ab_channel=NicholasRenotte)

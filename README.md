
# Deep Face Detection with Tensorflow and Python

I've been learning Deep Learning by doing projects like this. In this project, I took images of myself from webcam and then performed data augmentation to make the final dataset of over five thousand images.  Following this, I utilized TensorFlow's functional API to construct a model specifically designed for face detection within these images.








## Dataset
* Collected images from webcam using OpenCV
* Annotated those images with LabelMe
* Performed augmentation using Albumentations
## Build Deep Learning Model using the Functional API
* Downloaded VGG16(without the fully connected top layers) to build the model based on it.
* Built two separate branches on top of the VGG16 model:
        
    1. Classification Branch: Detects whether a face is present in the image.

    * Applied GlobalMaxPooling2D to reduce spatial dimensions.
    * Added a Dense layer with 2048 nodes and 'relu' activation function.
    * Final output layer is a Dense layer with one node and 'sigmoid' activation function.

    2. Bounding Box Branch: Locates the face in the image.
    * Applied another GlobalMaxPooling2D layer to reduce spatial dimensions.
    * Added a Dense layer with 2048 nodes and 'relu' activation function.
    * Final output layer is a Dense layer with four nodes (representing bounding box coordinates) and 'sigmoid' activation function.

* The two branches work together within a single model, allowing the system to both detect and locate a face within a given image. This is a multi-task learning approach in deep learning.
## Define Losses and Optimizers
* Defined Optimizer and Learning Rate
* Created Localization Loss and Classification Loss
## Train Neural Network
* Created custom Model class
* Trained the model over 10 epochs, allowing it to iteratively learn and optimize its performance on the detection and localization tasks.
## Making Predictions
* Predictions were made on the test set, evaluating the model's performance on unseen data.

* The model was further tested for real-time face detection using a webcam, demonstrating its capability to work with live video input.
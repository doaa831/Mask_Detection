# Mask_Detection
The Face Mask Detection System relies on the concepts of Computer vision and Deep learning using 
OpenCV and Tensorflow / Keras in order to detect face masks in still images as well as in real-time
video streams. So this system can be used in real-time applications that require face mask detection 
for safety purposes due to the Covid-19 virus outbreak. This project can be combined with integrated
systems for application in airports, railway stations, offices, schools and public places to ensure
that public safety guidelines are followed.

# Dataset
The dataset used can be downloaded here - [Click to Download ](https://drive.google.com/file/d/1NxxBwcPipK28TwKlpVKZSRXkvO-Twi_V/view?usp=sharing)

This dataset consists of 3835 images that fall into two categories:

  - with_mask: 1916 images
   
  - without_mask: 1919 images

The photos used were actual photos of faces wearing masks. Pictures were collected from the following sources:

  - [x] Bing Search API [(See Python script)](https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/search.py)
  - [x] Kaggle datasets
  - [x] RMFD dataset[(See here)](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)

# Model 
The face mask detector did not use any masking image data set. The model is accurate, and since we used the
**ResNet50 Architecture** without layers categorized as the primary model for feature extraction from images,
it is also computationally efficient and thus makes it easy to deploy the model to embedded systems.

- **ResNet50** is a pre-trained Keras model with the feature of letting you use weights that have already
  been calibrated to make predictions. In this case we're using weights from Imagenet and the network is ResNet50.
      
      Include_top = False 
      
  allows you to extract features by removing the last dense layers. This allows us to control form output and input.
   
      input_tensor = input (format = (224, 224, 3))
      base_model = ResNet50 (include_top = False, weights = 'imagenet', input_tensor = input_tensor) 
      
- The starting point is very helpful since we have weights already used to classify images but since we're using
  them in a completely new dataset, adjustments are needed. Our goal is to build a model that has high accuracy
  in its classification. This indicates how you will use previously trained layers of a model.
  We already have too many parameters due to the number of ResNet50 layers but we have calibration weights.

- We can choose to **freeze** these layers (as much as you can) so that these values do not change,
  this way saving time and computational cost.
  In this case, we are "freezing" all ResNet50 layers. The way to do this in Keras is by using:
  
      For the layer in base_model.layers:
         layer.trainable = false   
         
- Later on, we need to link our previously tested pattern with the new layers of our model.
  We used the **GlobalAveragePooling2D** layer to link the dimensions of the previous layers to the new layers.
  Using only **GlobalAveragePool2D layer, dense layer with relu and dense layer with softmax**,
  we can perform form closing and start the classification procedure.
  
- [x] **Optimization methods**: We tested it with 100 epochs using **RMSprop** to obtain the result.

- [x] **You can download Our Model from here:[(Click here)](https://drive.google.com/file/d/1VdBF9ZC6WGJ6dfSiH3rOEMzDFhaMf4pb/view?usp=sharing)**


# How to Use
To use this project on your system, follow these steps:

1- Clone this repository onto your system by typing the following command on your Command Prompt:

    git clone https://github.com/dodo295/Mask_Detection

followed by:

    cd Mask_Detection
    
 2- Ensure that you have all the required libraries used in all files.
   In case a library is missing, download it using pip, by typing this on your Command Prompt:
      
    pip install 'library name'

Replace 'library-name' by the name of the library to be downloaded.
    
3- **Train your Model** by typing the following commands on your Command Prompt:
      
    python loader.py Train_Model.py
    
4- Run Mask_Detection_Image.py by typing the following command on your Command Prompt:
    
    python Mask_Detection_Image.py --image Your_test.jpg
    
5- To open your webcam and discover if there is a mask or not! Writing:

    python Mask_Detection_Webcam.py 
   
# Results
- We got **100% accuracy in the training set** and **99.74% verification set** with 100 epochs.
![grab-landing-page](https://github.com/dodo295/Mask_Detection/blob/main/train_Acc.png)
![grab-landing-page](https://github.com/dodo295/Mask_Detection/blob/main/train_loss.png)

- Debate over the use of freezing continues in the previously tested model.
It reduces calculation time, reduces overuse but reduces accuracy.When the new data set is very
different from the data set used for training, it may be necessary to use more layers for modification.


- When selecting hyperparameters, it is important to impart learning to use a low learning rate to take
advantage of ready-made model weights. This selection as an optimizer option (SGD, Adam, RMSprop)
will affect the number of durations required to successfully obtain a trained model.

# Examples
![grab-landing-page](https://github.com/dodo295/Mask_Detection/blob/main/Output1.png)
![grab-landing-page](https://github.com/dodo295/Mask_Detection/blob/main/Output2.png)

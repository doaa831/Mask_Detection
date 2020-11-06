# Mask_Detection
The Face Mask Detection System relies on the concepts of Computer vision and Deep learning
using OpenCV and Tensorflow / Keras in order to detect face masks in still images
as well as in real-time video streams.

# :point_right: Dataset
The dataset used can be downloaded here - [Click to Download ](https://drive.google.com/file/d/1NxxBwcPipK28TwKlpVKZSRXkvO-Twi_V/view?usp=sharing)

This dataset consists of 3835 images that fall into two categories:

  - with_mask: 1916 images
   
  - without_mask: 1919 images

The photos used were actual photos of faces wearing masks. Pictures were collected from the following sources:

  - [x] Bing Search API [(See Python script)](https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/search.py)
  - [x] Kaggle datasets
  - [x] RMFD dataset[(See here)](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)
  
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
   

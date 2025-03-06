# Sign Detection (Text and Audio output)

## Abstract
Sign language is one of the oldest and most natural form of language for communication, hence we have come up with a real time method using neural networks for finger spelling based American sign language. Automatic human gesture recognition from camera images is an interesting topic for developing vision. We propose a convolution neural network (CNN) method to recognize hand gestures of human actions from an image captured by camera. The purpose is to recognize hand gestures of human task activities from a camera image. The position of hand and orientation are applied to obtain the training and testing data for the CNN. 

## Introduction
American Sign Language is a predominant sign language Since the only disability D&M people have been communication related and they cannot use spoken languages hence the only way for them to communicate is through sign language. Communication is the process of exchange of thoughts and messages in various ways such as speech, signals, behavior and visuals. Deaf and Dumb (D&M) people make use of their hands to express different gestures to express their ideas with other people. Gestures are the nonverbally exchanged messages and these gestures are understood with vision. This nonverbal communication of deaf and dumb people is called sign language. In our project we basically focus on producing a model which can recognise Fingerspelling based hand gestures in order to form a complete word by combining each gesture. 
#### ASL Signs

![App Screenshot](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRIcQ_GDGoz4vxbyatGMqNRZjEFEDdmM7JZmwGfkZfNLFf_eh4h8fu-WJUrA5njIAxiFSc&usqp=CAU)

## Objective
To create a computer software and train a model using CNN which takes an image of hand gesture of American Sign Language and shows the output of the particular sign language in text format converts it into audio format. 

## Implementation
### Data Acquasition
Different glove-based approaches can be used to extract information.

It uses electromechanical devices to provide exact hand configuration, and position. 

But it is expensive and not user friendly. 

In vision-based methods, the computer webcam is the input device for observing the information of hands and/or fingers. 

The Vision Based methods require only a camera, thus realizing a natural interaction between humans and computers without the use of any extra devices, thereby reducing costs

Here we have taken all the gestures as input through a webcam and stored them in appropriate folders.

File to take input through a webcam:
[data_collection_binary](https://github.com/sridharpatwari/Sign-Detection/blob/main/Files/data_collection_binary.py)

### Data pre-processing and Feature extraction
In this approach for hand detection, firstly we detect hand from image that is acquired by 
webcam and for detecting a hand we used media pipe library which is used for image 
processing. So, after finding the hand from image we get the region of interest (Roi) then 
we cropped that image using OpenCV library. We have collected images of different 
signs of different angles  for sign letter A to Z.

Link to Dataset: 
[Dataset](https://github.com/sridharpatwari/Sign-Detection/tree/main/Dataset/images1)

In this method there are many loop holes like your hand must be ahead of clean soft 
background and that is in proper lightning condition then only this method will give 
good accurate results but in real world we don’t get good background everywhere and 
we don’t get good lightning conditions too. 
So to overcome this situation we try different approaches then we reached at one 
interesting solution in which firstly we detect hand from frame using mediapipe and 
get the hand landmarks of hand present in that image then we draw and connect those 
landmarks in simple white image

Mediapipe Landmarking System

![Alt Text](https://mediapipe.dev/images/mobile/hand_landmarks.png)

Link to draw landmarks on image:
[Skeleton_images](https://github.com/sridharpatwari/Sign-Detection/blob/main/Files/skeleton_images.py)


Dataset after landmarking on the hand:
[Dataset](https://github.com/sridharpatwari/Sign-Detection/tree/main/Dataset/Skeleton_Images)

Now we will get this landmark points and draw it in plain white background using opencv 
library.


Link to draw skeleton images on the white image:
[Skeleton_mapping](https://github.com/sridharpatwari/Sign-Detection/blob/main/Files/skeleton_mapping.py)

Dataset after mapping the landmarks:
[Dataset](https://github.com/sridharpatwari/Sign-Detection/tree/main/Dataset/AtoZ_3.1)

### Gesture Classification 
####  Convolutional Neural Network (CNN)
CNN is a class of neural networks that are highly useful in solving computer vision problems. 
They found inspiration from the actual perception of vision that takes place in the visual 
cortex of our brain. They make use of a filter/kernel to scan through the entire pixel values of 
the image and make computations by setting appropriate weights to enable detection of a 
specific feature. CNN is equipped with layers like convolution layer, max pooling layer, 
flatten layer, dense layer, dropout layer and a fully connected neural network layer.  Unlike 
regular Neural Networks, in the layers of CNN, the neurons are arranged in 3 dimensions: 
width, height, depth. The neurons in a layer will only be connected to a small region of the 
layer (window size) before it, instead of all of the neurons in a fully-connected manner. 
Moreover, the final output layer would have dimensions(number of classes), because by the 
end of the CNN architecture we will reduce the full image into a single vector of class scores.

![Alt Text](https://ik.imagekit.io/upgrad1/abroad-images/imageCompo/images/unnamed8PDPDZ.png?pr-true)

#### Convolutional Layer: 
In convolution layer I have taken a small window size [typically of length 5*5] that 
extends to the depth of the input matrix. 
The layer consists of learnable filters of window size. During every iteration I slid the 
window by stride size [typically 1], and compute the dot product of filter entries and 
input values at a given position. 
As I continue this process well create a 2-Dimensional activation matrix that gives the 
response of that matrix at every spatial position. 
That is, the network will learn filters that activate when they see some type of visual 
feature such as an edge of some orientation or a blotch of some colour.

#### Pooling Layer:
We use pooling layer to decrease the size of activation matrix and ultimately reduce 
the learnable parameters. 
There are two types of pooling
##### 1. Max Pooling:
In max pooling we take a window size [for example window of size 2*2], and 
only taken the maximum of 4 values. 
 Well lid this window and continue this process, so well finally get an 
activation matrix half of its original Size.
##### 2. Average Pooling:
In average pooling we take average of all Values in a window.

![Alt Text](https://www.kdnuggets.com/wp-content/uploads/arham_diving_pool_unraveling_magic_cnn_pooling_layers_4.png)

####  Fully Connected Layer:
In convolution layer neurons are connected only to a local region, while in a fully connected 
region, well connect the all the inputs to neurons.
The preprocessed 180 images/alphabet will feed the keras CNN model.  
Because we got bad accuracy in 26 different classes thus, We divided whole 26 different 
alphabets into 8 classes in which every class contains similar alphabets:
All the gesture labels will be assigned with a probability. The label with the highest 
probability will treated to be the predicted label. So when model will classify [aemnst] in one 
single class using mathematical operation on hand landmarks we will classify further into 
single alphabet a or e or m or n or s or t.

![Alt Text](https://indiantechwarrior.com/wp-content/uploads/2023/08/Hidden-Layer-1-1024x603-min.jpg)

### Text and Speech Translation
The model translates known gestures into words. we have used pyttsx3 library to convert the 
recognized words into the appropriate speech. The text-to-speech output is a simple 
workaround, but it's a useful feature because it simulates a real-life dialogue.

## Conclusion
Finally, we are able to predict any alphabet[a-z] with 88% Accuracy (with and without clean 
background and proper lightning conditions) through our method. And if the background is 
clear and there is good lightning condition then we got even above 90% accurate results. In 
Future work we will make one android application in which we implement this algorithm for 
gesture predictions. 

![Image](https://github.com/user-attachments/assets/454023c4-b1a8-4950-a3cd-21bc987f6d0a)
 

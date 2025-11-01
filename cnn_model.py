"""
*****************************************************************************************
This model was built by: Djaid Walid

__________________________________________________________________________________________________
                                   Contacts                                                      |
__________________________________________________________________________________________________
Github     | https://github.com/waliddj                                                          |
Linkedin   | www.linkedin.com/in/walid-djaid-375777229                                           |
Instagram  | https://www.instagram.com/d.w.science?igsh=MWlnMmNpOTM2OW0xaA%3D%3D&utm_source=qr   |
__________________________________________________________________________________________________

Dataset used to train this model is : Astrophysical_Objects_Image_Dataset_Maxia_E.
Link to the dataset: https://www.kaggle.com/datasets/engeddy/astrophysical-objects-image-dataset/data
*****************************************************************************************
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
import numpy as np
import pathlib
import os

# ************* change the path ***************
# Laod the dataset
data_dir_path = 'C:/Users/walid/Desktop/astro_dataset_maxia/astro_dataset_maxia'

# Split the training, test and validation data
train_dir = 'C:/Users/walid/Desktop/astro_dataset_maxia/astro_dataset_maxia/training'
test_dir = 'C:/Users/walid/Desktop/astro_dataset_maxia/astro_dataset_maxia/test'
valid_dir = 'C:/Users/walid/Desktop/astro_dataset_maxia/astro_dataset_maxia/validation'
"""
for dirpath, dirnames,filenames in os.walk(data_dir_path):
    print(f"there are {len(dipiprnames)} directories and {len(filenames)} imagies in {dirpath}")
"""

# Get the class names from the dataset
data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))

# ******** Preprocessing the data ********

# Augmentation of the training data using tf.keras.preprocessing.image.ImageDataGenerator
"""
train_datagen = ImageDataGenerator(rescale=1./255,
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                horizontal_flip=True)"""

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20)
#Normalization of the test and validation data
test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Load data from directories and turn them into batches
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224), # resizing the data
    batch_size=32,
    class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
valid_data = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ******** Build the CNN model ********
tf.random.set_seed(42)
# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=10,kernel_size=3, input_shape=(224,224,3), activation='relu'),
    tf.keras.layers.Conv2D(filters=10,kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Conv2D(filters=10,kernel_size=3, activation='relu'),
    tf.keras.layers.Conv2D(filters=10,kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])
# Compile the model
model.compile(
    loss = 'categorical_crossentropy',
    optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
# fit the model to the training data and track the history of the training
history = model.fit(train_data,
                    epochs=8,
                    steps_per_epoch=len(train_data),
                    validation_data=valid_data,
                    validation_steps=len(valid_data))

# ******** Evaluate the model ********
model.evaluate(test_data) # Evaluate the model on the test data

import pandas as pd

pd.DataFrame(history.history).plot() # plot the Accuracy of the model and the loss functions of both the training data and validation data
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

# ******** Save the model ********
tf.keras.models.save_model(model,"C:/Users/walid/Desktop/cnn_model_2.keras")

"""
******************************** Model Testing ****************************************

**************** MODEL 1 (Baseline) ****************
** Data Preprocessing:
- Training data was augmented by Normalization, rotate the training images by 20 degrees, the width and height were shifted by a value of 0.2, 
and the images were horizontally flipped.
- Test and validation data were Normalized.

Architecture:
- (2 Conv2D layer followed by a MaxPool2D() layer) x 2
- Activation method used for the conv layers is ReLU
- Activation metho used for the output layer is SoftMax


Model Accuracy&Loos:

Accuracy = 75%
Validation accuracy = 80%
Test acccuracy = 83%
Loss = 0.8

******************************************
"""

"""
**************** MODEL 2 ****************
Architecture:
- Same as the MODEL 1, difference is in MaxPool2D(2) function


Model Accuracy&loss

Accuracy = 77%, loss= 0.67
Validation accuracy = 83%,  loss= 0.66
Test accuracy = 83%,  loss= 0.49

> Note:
The MaxPool2D value does not affect the improvement of the model
*****************************************
"""

"""
**************** MODEL 3 ****************
Architecture:
- Same as the MODEL 2, difference is in number of epochs (6->12 epochs)


Model Accuracy&loss:

Accuracy = 82%, loss= 0.59
Validation accuracy = 84%,  loss= 0.67
Test acccuracy = 82%,  loss= 0.519

> NOTE:
A slight difference between MODEL 2 and MODEL 3 in the Validation accuracy (Model 2 : 83% < Model 3 : 84%).
Moreover, the model 3 Accuracy is higher than the previous model's one (Model2: 77% < model 3: 82%). However,
the test accuracy of the 2nd model is higher due to an overfitting.

> Conclusion:
The optimal number of epochs is 6.
***************************************
"""

"""
**************** MODEL 4 ****************
Architecture:
- Learning rate = 0.01


Model Accuracy&loss:

Accuracy = >10%, loss= <2
Validation accuracy = >10%,  loss= <2
Test acccuracy = >10%,  loss= <2

>Note:
The MODEL 4 accuracy is less than 10% due to the high learning rate that cause oferfitting.

>Conclusion:
The optimial Learning rate is 0.001 (default)
*****************************************
"""

"""
**************** MODEL 5 ****************
Architecture:
- Increasing the number of conv and maxpool layers (Adding two more layers of conv and one of maxpool). 


Model Accuracy&loss:

Accuracy = 72%, loss= 0.8 
Validation accuracy = 76%,  loss= 0.79
Test acccuracy = 76%,  loss= 0.67

>Note:
The test accuracy drops from (in the model 2) 83% to 76% due to overfittig

>Conclusion:
Adding more layers lead to model overfitting
*****************************************
"""

"""
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(224, 224, 3)),
    Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
    
    Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
    
    Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(class_names), activation='softmax')
], name='astro_classifier')

**************** MODEL 6 ****************
Architecture:
- 2 Conv layers followed by a MaxPool2D(2)
- 1 Conv layer  followed by a MaxPool2D(2)
- 1 Conv layer followed by a MaxPool2D(2)
- Flatten layer
- 2 Dense layers with 128 hidden uinits and 64 hidden unitis respectivelly with a ReLU activation
- output layer (output_shape (len(class_names)) with a softmax activation method
 - number of epochs 10
 

Model Accuracy&loss:

Accuracy = 79%, loss= 0.57
Validation accuracy = 82%,  loss= 0.61
Test acccuracy = 81%,  loss= 0.49

>Note:
Model 2 > Model 6 

>Conclusion:

*****************************************

**************** MODEL 7 ****************
Architecture:
- same as model 2.
- adding some randomness to the data.
- number of epochs 10

Model Accuracy&loss:

Accuracy = %, loss= 
Validation accuracy = %,  loss= 
Test acccuracy = %,  loss= 

>Note:
the loss function of the validation data drop from 2 to 0.2 (in the 8th epoch) however it stared increasing just after the 8th epoch
> Conclusion:
The optimial number of training epochs is 8.
*****************************************


**************** MODEL 8 ****************
Architecture:
- same as model 7.
- number of training epochs= 8 (10->8)
- No data augmentation only Normalization

Model Accuracy&loss:

Accuracy = 89%, loss= 0.15
Validation accuracy = 88%,  loss= 0.7
Test acccuracy = 89%,  loss= 0.42

>Note:
the loss function of the validation data drop from 2 to 0.7

> Conclusion:
The optimial number of training epochs is 8.
*****************************************



**************** MODEL 9 ****************
Preprocessing data:
- Training data was augmented by Normalization, rotate the training images by 20 degrees.
- Test and validation data were Normalized.

Architecture:
- same as model 8

Model Accuracy&loss:

Accuracy = 92%, loss= 0.24
Validation accuracy = 88.75%,  loss= 0.45
Test acccuracy =  89.69%,  loss= 0.43

>Note:
A slight increase of the test accuracy compared to the previous model.

> Conclusion:
Rotating the image does not lead to a noticeable improvement of the model.
*****************************************

*****************************************************************************************
"""





"""
******************************** FINAL MODEL (MODEL 9)  ********************************
** Data Preprocessing:
- Training data was augmented by Normalization, rotate the training images by 20 degrees.
- Test and validation data were Normalized.

** Architecture:
- Input layer Conv2D with input_shape(224,224,3), 10 filters, kernel size of 3
- Conv2D layer with same parameters as the input layer without the input shape. 
- MaxPool() layer.
- 2 more Conv2D with same parameters followed by a MaxPool() layer.
- Fllaten layer followed by an output layer (Dense layer) with an output shape of len(class_names), and a softmax activation method.

** Model Accuracy&loss:
______________________________________________
Accuracies                    |      Losses   |
______________________________________________
Accuracy = 92%                |   loss= 0.24  |
Validation accuracy = 88.75%  |   loss= 0.45  |
Test acccuracy = 89.69        |   loss= 0.43  |
______________________________________________

*****************************************************************************************
"""


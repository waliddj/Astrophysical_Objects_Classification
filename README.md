# Astrophysical_Objects_Classification
Astrophysical_Objects_Classification is a Convolutional Neural Network (CNN) multi-class classification model for astrophisical objects, such as: planets, galaxies, black holes...etc

# Dataset:

the dataset used for training this model is [Astrophysical_Objects_Image_Dataset_Maxia_E](https://www.kaggle.com/datasets/engeddy/astrophysical-objects-image-dataset/data).

## Dataset structure:
The dataset is divided into three main directories:

train/
validation
test
Each of these directories contains subfolders corresponding to 12 astrophysical object classes:

- ```asteroid```
- ```black_hole```
- ```earth```
- ```galaxy```
- ```jupiter```
- ```mars```
- ```mercury```
- ```neptune```
- ```pluto```
- ```saturn```
- ```uranus```
- ```venus```
- 
### Example structure:
train/

├── asteroid/

├── black_hole/

├── earth/

├── galaxy/

├── …

validation/

├── asteroid/

├── black_hole/

├── …

test/

├── asteroid/

├── black_hole/

├── …

# Code architecture:

```python
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
import numpy as np
import pathlib
import os
import pandas as pd
```

## 1. Load the data set:

### Load the dataset
```python
data_dir_path = 'C:/Users/walid/Desktop/astro_dataset_maxia/astro_dataset_maxia'
```
### Split the training, Validation, and test data
```python
train_dir = 'C:/Users/walid/Desktop/astro_dataset_maxia/astro_dataset_maxia/training'
test_dir = 'C:/Users/walid/Desktop/astro_dataset_maxia/astro_dataset_maxia/test'
valid_dir = 'C:/Users/walid/Desktop/astro_dataset_maxia/astro_dataset_maxia/validation'
```
### Get the classnames
```python
data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
```

## 2. Preprocessing the data:

### Data augmentation and normalization:
Normalize and rotate the training data using [```tensorflow.kers.preprocessing.image.ImageDataGenerator```](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
```python
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20)
```
Normalize the test and validation data
```python
test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
```
> **Note:** Data normalization means ajusting the pixel values on a scale between 0 and 1. This technique used to improve the training of deep neural networks by stabilizing the learning process.

Load data from directories and turn them into batches
```python
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
```

## 3. Build the CNN model:
### Model architecture:
- Input layer ```Conv2D``` with ```input_shape(224,224,3)```, 10 filters, kernel size of 3, with a ```ReLU``` activation method.
- Conv2D layer with same parameters as the input layer without the input shape, with a ```ReLU``` activation method.
- ```MaxPool() layer```.
- 2 more ```Conv2D``` with same parameters and activation method, followed by a ```MaxPool()``` layer.
- ```Fllaten``` layer followed by an output layer (```Dense layer```) with an output shape of ```len(class_names)```, with a ```softmax``` activation method.
  
Build the model
```python
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
```
Compile the model
```python
model.compile(
    loss = 'categorical_crossentropy',
    optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
```
Fit the model to the training data and track the history of the training
```python
history = model.fit(train_data,
                    epochs=8,
                    steps_per_epoch=len(train_data),
                    validation_data=valid_data,
                    validation_steps=len(valid_data))
```

## 4. Evaluate the model:
Evaluate the model on the test data using the function ```evaluate()```
```p
model.evaluate(test_data)
```
Plot the Accuracy of the model and the loss functions of both the training data and validation data
```python
pd.DataFrame(history.history).plot() 
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

```
______________________________________________
Accuracies                    |      Losses   |
______________________________________________
Accuracy = 92%                |   loss= 0.24  |
Validation accuracy = 88.75%  |   loss= 0.45  |
Test acccuracy = 89.69        |   loss= 0.43  |
______________________________________________


## 5. Save the model
Save the model using [```tf.keras.models.save_model()```](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model)
```p
tf.keras.models.save_model(model,"path")
```

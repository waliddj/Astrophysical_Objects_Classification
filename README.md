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
data_dir_path = 'C:/Users/walid/Desktop/astro_dataset_maxia/astro_dataset_maxia' # path to the dataset
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
| Metric                     | Accuracy     | Loss     |
|----------------------------|--------------|----------|
| Training                   | 92%          | 0.24     |
| Validation                 | 88.75%       | 0.45     |
| Test                       | 89.69%       | 0.43     |

## 5. Save the model
Save the model using [```tf.keras.models.save_model()```](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model)
```p
tf.keras.models.save_model(model,"path")
```
## **Load the model and use it to predict**
### Load the model using [```tf.keras.models.load_model```](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model)
```p
model = tf.keras.models.load_model("C:/Users/walid/Desktop/cnn_model.keras")
```
Load the class names from the training directory 
```python
train_dir = 'C:/Users/walid/Desktop/astro_dataset_maxia/astro_dataset_maxia/training'
data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
```

### Plot the input image of an astrophysical object with the prediction and the confidence rate
Load the input image and resize it and Normalize the pixel values
```python
def load_and_prep_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [img_shape, img_shape])
    img = tf.cast(img, tf.float32) / 255.0
    return img
```
Predict the astrophysical object and plot the inputed image with it's predication and the confidence rate
```python
def pred_and_plot(model, filename, class_names, img_shape=224, top_k=3):
    # Load image
    img = load_and_prep_image(filename, img_shape)

    # Predict: shape (1, C) for multiclass or (1, 1) for binary
    out = model.predict(tf.expand_dims(img, axis=0), verbose=0)[0]

    # Convert to probabilities depending on output shape
    if out.ndim == 0:
        # very unusual, but convert a scalar logit
        probs = tf.nn.softmax([out]).numpy()
    elif out.shape[-1] == 1:  # binary case (sigmoid or logits)
        # If model used sigmoid, 'out' is already p1 in [0,1].
        # If model used logits, apply sigmoid; applying it again is harmless if already prob.
        p1 = tf.sigmoid(out).numpy().squeeze()
        probs = np.array([1.0 - p1, p1])
        # Ensure class_names has length 2 in this case.
        if len(class_names) != 2:
            # Optional: define your two labels explicitly
            class_names = np.array(['class0', 'class1'])
    else:
        # Multiclass case
        # If they already sum ~1 and are non-negative, treat as probs.
        if np.all(out >= 0) and np.isclose(np.sum(out), 1.0, atol=1e-3):
            probs = out
        else:
            probs = tf.nn.softmax(out).numpy()

    # Get predicted class and confidence
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    # Plot image with prediction
    plt.imshow(img.numpy())
    plt.title(f"Prediction: {pred_class}, confidence: {confidence:.2%}")
    plt.axis(False)
    plt.show()

    # Print top-k for inspection
    top_idx = np.argsort(probs)[::-1][:min(top_k, len(probs))]
    print("Top predictions:")
    for i in top_idx:
        print(f"  {class_names[i]}: {probs[i]:.2%}")
```
```Input```![earth](https://github.com/user-attachments/assets/df55923a-d2b5-4d87-a8cf-9b6ad65da195)

```Output```: 
<img width="640" height="480" alt="earth_pred" src="https://github.com/user-attachments/assets/423bdd3e-6400-45e5-8b31-0fcdf1e82e9f" />


```Input```<img width="1810" height="1639" alt="Jupiter" src="https://github.com/user-attachments/assets/40836694-fa3f-4dc7-8a1c-9f366be87a84" />

```Output```<img width="640" height="480" alt="jup_pred" src="https://github.com/user-attachments/assets/c30c73fe-daef-4489-bb53-d354f1ee9734" />


```Input```![gal](https://github.com/user-attachments/assets/6d2d9797-6d12-443f-8011-76f804365efd)

```Output```<img width="640" height="480" alt="gal_pred" src="https://github.com/user-attachments/assets/35c1a736-c797-441b-a226-2442b29998de" />


```Input```![48 Black Hole](https://github.com/user-attachments/assets/a3eb500b-4a80-4c94-812b-fa086c5a8b22)

```Output```<img width="640" height="480" alt="black_hole_pred" src="https://github.com/user-attachments/assets/0d31cdd5-27dd-42e8-88f7-3ffad2abd7cb" />


# Appendix:
>The reason I chose that model architecture is that I have done multiple tests, builing and adjusting the model parameters and the data augmentation.

## Model 1 (Baseline):
### Data Preprocessing:
- Training data was augmented by Normalization, rotate the training images by 20 degrees, the width and height were shifted by a value of 0.2, 
and the images were horizontally flipped.
- Test and validation data were Normalized.
> **Note:** The training augmentation parameters were taken from a previous project.
### Architecture:
- (2 Conv2D layer followed by a MaxPool2D() layer) x 2
- Activation method used for the conv layers is ReLU
- Activation metho used for the output layer is SoftMax


### Model evaluation:
| Metric                     | Accuracy     | Loss     |
|----------------------------|--------------|----------|
| Training                   | 75%          |  0.8     |
| Validation                 | 80%          | 1.2     |
| Test                       | 83%          | 1.03     |

## Model 2:
### Data Preprocessing:
- Same as the MODEL 1
### Architecture:
- Same as the MODEL 1, difference is in MaxPool2D(3) function
### Model evaluation
|Metric|Accuracy|Loss|
|----------------------------|--------------|----------|
|Accuracy  |77% | 0.67|
|Validation accuracy | 83% |   0.66|
|Test accuracy | 83%|   0.49|

> **Note:**
The MaxPool2D value does not affect the improvement of the model


## Model 3:
### Architecture:
- Same as the MODEL 2, difference is in number of epochs (6->12 epochs)
### Model evaluation:
|Metric|Accuracy|Loss|
|----------------------------|--------------|----------|
|Accuracy | 82%|  0.59|
|Validation accuracy| 84%|  0.67|
|Test acccuracy | 82%|   0.519|
> **NOTE:**
A slight difference between MODEL 2 and MODEL 3 in the Validation accuracy (Model 2 : 83% < Model 3 : 84%).
Moreover, the model 3 Accuracy is higher than the previous model's one (Model2: 77% < model 3: 82%). However,
the test accuracy of the 2nd model is higher due to an overfitting.

> ***Conclusion:***
The optimal number of epochs is 6.


## Model 4:
### Architecture:
- Learning rate = 0.01
## Model evaluation:
|Metric|Accuracy|Loss|
|----------------------------|--------------|----------|
|Accuracy | 12%| <2|
|Validation accuracy | 10%| <2|
|Test acccuracy | 9.8%|<2|

>**Note:**
The MODEL 4 accuracy is less than 10% due to the high learning rate that cause oferfitting.

>***Conclusion:***
The optimial Learning rate is 0.001 (default)


## Model 5:
### Architecture:
- Increasing the number of conv and maxpool layers (Adding two more layers of conv and one of maxpool). 
### Model evaluation:
|Metric|Accuracy|Loss|
|----------------------------|--------------|----------|
|Accuracy | 72%| 0.8 |
|Validation accuracy| 76%| 0.79|
|Test acccuracy | 76%| 0.67|

>**Note:**
The test accuracy drops from (in the model 2) 83% to 76% due to overfittig

>***Conclusion:***
Adding more layers lead to model overfitting


## Model 6:
### Architecture:
> Model architecture inspired from [CNN-astronomical classification](https://www.kaggle.com/code/devanshshukla123/cnn-astronomical-classification#Step-10:-Test-Predictions)
- 2 Conv layers followed by a MaxPool2D(2)
- 1 Conv layer  followed by a MaxPool2D(2)
- 1 Conv layer followed by a MaxPool2D(2)
- Flatten layer
- 2 Dense layers with 128 hidden uinits and 64 hidden unitis respectivelly with a ReLU activation
- output layer (output_shape (len(class_names)) with a softmax activation method
 - number of epochs 10
 ### Model evaluation:
|Metric|Accuracy|Loss|
|-----|---------|----|
|Accuracy | 79%| 0.57|
|Validation accuracy | 82%| 0.61|
|Test acccuracy |81%| 0.49|

>**Note:**
Model 2 > Model 6

## Model 7:
### Architecture:
- same as model 2.
- adding some randomness to the data.
- number of epochs 10

 ###Model evaluation:
|Metric|Accuracy|Loss|
|----------------------------|--------------|----------|
|Accuracy | 75%| 0.8|
|Validation accuracy | 83.9%| 0.4|
|Test acccuracy |84%| 0.51|

>**Note:**
the loss function of the validation data drop from 2 to 0.4 (in the 8th epoch) however it stared increasing just after the 8th epoch
> ***Conclusion:***
The optimial number of training epochs is 8, adding randomness leads to an improvement of the model.


## Model 8:
### Architecture:
- same as model 7.
- number of training epochs= 8 (10->8)
- No data augmentation only Normalization

### Model evaluation:
|Metric|Accuracy|Loss|
|----------------------------|--------------|----------|
|Accuracy | 89%| 0.15|
|Validation accuracy = 88%| 0.7|
|Test acccuracy = 89%| 0.42|

>**Note:**
the loss function of the validation data drop from 2 to 0.7

> ***Conclusion:***
The optimial number of training epochs is 8, no data augmentation leades to an improvement of the model only if the input image is clear.

## Model 9
### Data Preprocessing:
- Training data was augmented by Normalization, rotate the training images by 20 degrees.
- Test and validation data were Normalized.

### Architecture:
- Input layer Conv2D with input_shape(224,224,3), 10 filters, kernel size of 3
- Conv2D layer with same parameters as the input layer without the input shape. 
- MaxPool() layer.
- 2 more Conv2D with same parameters followed by a MaxPool() layer.
- Fllaten layer followed by an output layer (Dense layer) with an output shape of len(class_names), and a softmax activation method.

### Model evaluation:
| Metric                     | Accuracy     | Loss     |
|----------------------------|--------------|----------|
| Training                   | 92%          | 0.24     |
| Validation                 | 88.75%       | 0.45     |
| Test                       | 89.69%       | 0.43     |

>**Note:**
A slight increase of the test accuracy compared to the previous model.

>***Conclusion:***
Rotating the image does not lead to a noticeable improvement of the model specially when the input image is not clear.

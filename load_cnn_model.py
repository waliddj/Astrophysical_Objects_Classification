import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib

def load_and_prep_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [img_shape, img_shape])
    img = tf.cast(img, tf.float32) / 255.0
    return img

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

# ************* change the path ***************
model = tf.keras.models.load_model("C:/Users/walid/Desktop/cnn_model.h5")
model_2 = tf.keras.models.load_model("C:/Users/walid/Desktop/cnn_model_2.keras")
train_dir = 'C:/Users/walid/Desktop/astro_dataset_maxia/astro_dataset_maxia/training'
data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))

file_img = 'C:/Users/walid/Desktop/gal.jpeg'
file_img_2 = 'C:/Users/walid/Desktop/Jupiter.png'
file_img_3 = 'C:/Users/walid/Desktop/earth.jpg'
file_img_4 = 'C:/Users/walid/Desktop/48 Black Hole.jpg'
file_img_5 = 'C:/Users/walid/Desktop/Asteroid_Vesta-1.jpg'

pred_and_plot(model_2, file_img_5, class_names, img_shape=224, top_k=5)

"""
model_2 could predict the earth.jpg, confidence= 98.69%, where model predicted uranus
"""


from PIL import Image
from IPython.display import display
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from pathlib import Path

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

name_model = './models/best_my_model-21-05-22_222322'
reconstructed_model = keras.models.load_model(name_model)
reconstructed_model_pred = keras.models.Model(
    reconstructed_model.get_layer(
        name="image").input, reconstructed_model.get_layer(name="dense2").output
)
# Store arrays in memory as it's not a muvh big dataset


def generate_arrays(df, resize=True, img_height=50, img_width=200):
    """Generates image array and labels array from a dataframe.

    Args:
        df: dataframe from which we want to read the data
        resize (bool)    : whether to resize images or not
        img_weidth (int): width of the resized images
        img_height (int): height of the resized images

    Returns:
        images (ndarray): grayscale images
        labels (ndarray): corresponding encoded labels
    """

    num_items = len(df)
    images = np.zeros((num_items, img_height, img_width), dtype=np.float32)
    labels = [0]*num_items

    for i in range(num_items):
        img = cv2.imread(df["img_path"][i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if resize:
            img = cv2.resize(img, (img_width, img_height))

        img = (img/255.).astype(np.float32)
        label = df["label"][i]

        images[i, :, :] = img
        labels[i] = label

    return images, np.array(labels)


# Path to the data directory
data_dir = Path("./data/CAPTCHA_images/img")

# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))[:10000]
labels = [img.split(os.path.sep)[-1].split(".jpg")[0] for img in images]
characters = set(char for label in labels for char in label)

# Mapping characters to integers
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)
max_length = max([len(label) for label in labels])

# A utility function to decode the output of the network


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


data = {
    "img_path": "./data/CAPTCHA_images/img/x5no.jpg",
    "label": "x5no"
}
df_test = pd.DataFrame([data])
data_test, labels_test = generate_arrays(
    df_test, resize=True, img_height=60, img_width=160)

print(np.array([data_test.T]).shape)

print('PREDICT')
print(reconstructed_model_pred.summary())
preds = reconstructed_model_pred.predict(np.array([data_test.T]))
print('PREDICTED')
pred_texts = decode_batch_predictions(preds)
print(pred_texts)

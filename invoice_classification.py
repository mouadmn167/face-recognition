#!/usr/bin/env python
# coding:utf-8
"""
Name : author.py
Author : OBR01
Contact : oussama.brich@edissyum.com
Time    : 10/09/2020 15:56
Desc: this python show how to use the invoice classification model
 """

import os

# Set error/warning level to 3 (to much warnings displayed by Tensorflow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import  tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Prediction label list
'''
Every label must be changed with the right label 
keeping the same order 
Example : if "EXEMPLE_1"="TYPOLOGIE_1" then
"EXEMPLE_1" must be replaced with "TYPOLOGIE_1" in the same position (first position for this example)
'''
enabled          = False
iaPath           = '/invoice_classification/'
MODEL_PATH       = 'invoice_classification.model'
trainImagePath   = 'images'
predictImagePath = '/images/predict-images/'
confidenceMin    = 80

IMG_HEIGHT = 699
IMG_WIDTH = 495
MODEL_PATH = ''  # Tensorflow model path
LABELS_ORDERED_LIST = ["messi","mounaji","ronaldo"]



# Train and save tensorFlow model
def train(images_data_path):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        images_data_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32)

    autotune = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = train_ds.cache().prefetch(buffer_size=autotune)

    num_classes = 3

    input_shape = (IMG_HEIGHT, IMG_WIDTH)

    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(23, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(23, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(23, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dense(num_classes)
    ])
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15
    )

    # Save model
    model.save(MODEL_PATH)


def predict(path_image_test):
    # Load image using Keras
    img = keras.preprocessing.image.load_img(
        path_image_test, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )

    # Convert image to array
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Use Tensorflow model to get predoct list
    if os.path.exists(MODEL_PATH):
        print("hello")
        try:
            
            digit_model = tf.keras.models.load_model(MODEL_PATH)
            pred_probab = digit_model.predict(img_array)[0]
        except IndexError:
            return None
        return pred_probab
    


def predict_typo(pdf_path):
    image_path = get_pdf_first_page(pdf_path)
    pred_probab = predict(image_path)

    file_name = pdf_path.split("/")[-1].replace('pdf', 'jpg')
    image_path = PREDICT_IMAGES_PATH + file_name

    if pred_probab is not None:
        pred_index = list(pred_probab).index(max(pred_probab))
        typo = LABELS_ORDERED_LIST[pred_index]
        predict_op = tf.nn.softmax(pred_probab)
        pred_index = list(predict_op).index(max(predict_op))
        prediction_percentage = str(float('%2f' % int(predict_op[pred_index].numpy() * 100)))

        try:
            os.remove(image_path)
        except FileNotFoundError:
            pass

        return typo, prediction_percentage
    else:
        try:
            os.remove(image_path)
        except FileNotFoundError:
            pass
        return False, False



#train(trainImagePath)
print(predict("me.jpeg"))
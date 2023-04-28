import tensorflow as tf
import numpy as np
import zipfile

from settings import *

# load zip
with zipfile.ZipFile("./datasets/" + DATASET_NAME, 'r') as zip_ref:
    zip_ref.extractall("./extracts/" + DATASET_NAME)

# load in tensorflow dataset
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data = data_generator.flow_from_directory(
    "./extracts/" + DATASET_NAME,
    target_size=(512, 512),
    batch_size=BATCH_SIZE,
    class_mode='input')

# define embedding model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(512, 512, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(3, (3,3), activation=None, padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation=None, name="embedding")
])

# compile model
optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='mse')

# training model
model.fit(train_data, 
          epochs=EPOCHS)

# saving embedding
embedding = model.get_layer("embedding")
embedding_weights = embedding.get_weights()[0]
np.save("embedding.npy", embedding_weights)
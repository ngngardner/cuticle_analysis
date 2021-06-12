

from typing import List

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from .dataset import Dataset


class CNN():
    def __init__(self, data: Dataset):
        self.data = data
        self.name = 'CNN'
        self.num_classes = len(np.unique(data.labels))
        self.subimage_size = data.size
        self.path = f'./output/model_{data.size[0]}_{data.size[1]}'

        self.model = Sequential([
            layers.experimental.preprocessing.Rescaling(
                1./255,
                input_shape=(
                    self.subimage_size[0], self.subimage_size[1], 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes)
        ])

    def metadata(self) -> List[str]:
        return [
            f'Model Type: {self.name}'
        ]

    def train(self, epochs, test_size) -> None:
        train_x, test_x, train_y, test_y = train_test_split(
            self.data.subimages,
            self.data.labels,
            test_size=test_size)

        self.model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=['accuracy'])
        self.model.summary()
        self.epochs = epochs

        self.history = self.model.fit(
            train_x, train_y,
            validation_data=(test_x, test_y),
            epochs=epochs
        )

    def plot(self) -> None:
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig('cnn_output.test.png')

    def save_weights(self) -> None:
        self.model.save(self.path)

    def load_weights(self) -> None:
        self.model = keras.models.load_model(self.path)

    def predict(self, image: np.ndarray) -> np.ndarray:
        pred = self.model.predict(image)
        pred = np.argmax(pred, axis=1)
        return pred

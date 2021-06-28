

from typing import List

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from ..datasets import Dataset


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

    def train(self, epochs: int, n: int) -> None:
        train_x, train_y, test_x, test_y = self.data.stratified_split(n)

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

    def save_weights(self) -> None:
        self.model.save(self.path)

    def load_weights(self) -> None:
        self.model = keras.models.load_model(self.path)

    def predict(self, image: np.ndarray) -> np.ndarray:
        pred = self.model.predict(image)
        pred = np.argmax(pred, axis=1)
        return pred

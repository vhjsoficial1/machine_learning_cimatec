"""Definição da arquitetura da CNN."""
from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)

from module_classificador_imagens.config import IMAGE_SHAPE, NUM_CLASSES

# (filtros, dropout) de cada bloco convolucional
CONV_BLOCKS = [(32, 0.25), (64, 0.25), (128, 0.30)]


def build_model() -> keras.Model:
    """Constrói e compila a CNN (3 blocos conv + cabeça densa)."""
    model = keras.Sequential(name="cnn_cifar10")
    model.add(Input(shape=IMAGE_SHAPE))

    for filters, dropout in CONV_BLOCKS:
        for _ in range(2):
            model.add(Conv2D(filters, (3, 3), padding="same", activation="relu"))
            model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
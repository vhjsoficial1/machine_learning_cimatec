"""Pré-processamento de imagens e rótulos."""
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import img_to_array, load_img

from module_classificador_imagens.config import IMAGE_SHAPE


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Converte para float32 e normaliza os pixels para o intervalo [0, 1]."""
    return images.astype("float32") / 255.0


def flatten_labels(labels: np.ndarray) -> np.ndarray:
    """O CIFAR-10 entrega os rótulos como coluna (N, 1); aqui viram (N,)."""
    return labels.flatten()


def load_image_file(path: Path) -> np.ndarray:
    """Carrega uma imagem do disco pronta para o modelo: shape (1, 32, 32, 3)."""
    img = load_img(path, target_size=IMAGE_SHAPE[:2])
    array = normalize_images(img_to_array(img))
    return np.expand_dims(array, axis=0)
"""Classificacao de dados estruturados com Keras.

Exemplo de classificacao binaria com features numericas e categoricas.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import pandas as pd
import tensorflow as tf
from keras import layers


TARGET_FEATURE_NAME = "target"
NUMERIC_FEATURE_NAMES = [
    "age",
    "trestbps",
    "thalach",
    "oldpeak",
    "slope",
    "chol",
]
CATEGORICAL_FEATURE_NAMES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "ca",
    "thal",
]
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES


def prepare_data(dataframe):
    val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
    train_dataframe = dataframe.drop(val_dataframe.index)
    categorical_is_string = {
        feature_name: not pd.api.types.is_numeric_dtype(train_dataframe[feature_name])
        for feature_name in CATEGORICAL_FEATURE_NAMES
    }

    categorical_encoders = {}
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        feature_values = train_dataframe[feature_name]
        if categorical_is_string[feature_name]:
            vocabulary = sorted(feature_values.astype(str).unique())
            categorical_encoders[feature_name] = layers.StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=1,
                output_mode="one_hot",
            )
        else:
            vocabulary = sorted(feature_values.astype("int64").unique())
            categorical_encoders[feature_name] = layers.IntegerLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=1,
                output_mode="one_hot",
            )

    normalizers = {}
    for feature_name in NUMERIC_FEATURE_NAMES:
        normalizer = layers.Normalization(axis=-1)
        values = train_dataframe[feature_name].to_numpy(dtype="float32").reshape(-1, 1)
        normalizer.adapt(values)
        normalizers[feature_name] = normalizer

    def dataframe_to_dataset(frame, shuffle=False):
        frame = frame.copy()
        labels = frame.pop(TARGET_FEATURE_NAME).to_numpy(dtype="float32")
        features = {}
        for feature_name in FEATURE_NAMES:
            if categorical_is_string.get(feature_name, False):
                features[feature_name] = frame[feature_name].astype(str).to_numpy()
            else:
                features[feature_name] = frame[feature_name].to_numpy()
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        def encode_features(features, label):
            encoded = {}
            for feature_name in NUMERIC_FEATURE_NAMES:
                value = tf.cast(features[feature_name], tf.float32)
                encoded[feature_name] = tf.expand_dims(value, axis=-1)
            for feature_name in CATEGORICAL_FEATURE_NAMES:
                value = features[feature_name]
                if categorical_is_string[feature_name]:
                    value = tf.cast(value, tf.string)
                else:
                    value = tf.cast(value, tf.int64)
                encoded[feature_name] = tf.reshape(
                    categorical_encoders[feature_name](value), [-1]
                )
            return encoded, label

        dataset = dataset.map(encode_features)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(frame), seed=1337)
        return dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    return (
        dataframe_to_dataset(train_dataframe, shuffle=True),
        dataframe_to_dataset(val_dataframe),
        categorical_encoders,
        categorical_is_string,
        normalizers,
    )


def build_model(categorical_encoders, normalizers):
    model_inputs = {}
    processed_features = []

    for feature_name in NUMERIC_FEATURE_NAMES:
        feature_input = layers.Input(name=feature_name, shape=(1,), dtype=tf.float32)
        model_inputs[feature_name] = feature_input
        processed_features.append(normalizers[feature_name](feature_input))

    for feature_name in CATEGORICAL_FEATURE_NAMES:
        category_count = categorical_encoders[feature_name].vocabulary_size()
        feature_input = layers.Input(
            name=feature_name,
            shape=(category_count,),
            dtype=tf.float32,
        )
        model_inputs[feature_name] = feature_input
        processed_features.append(feature_input)

    all_features = layers.Concatenate()(processed_features)
    hidden = layers.Dense(32, activation="relu")(all_features)
    hidden = layers.Dropout(0.5)(hidden)
    output = layers.Dense(1, activation="sigmoid")(hidden)

    model = keras.Model(model_inputs, output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def predict_sample(model, sample, categorical_encoders, categorical_is_string):
    sample_inputs = {}
    for feature_name in NUMERIC_FEATURE_NAMES:
        sample_inputs[feature_name] = tf.constant(
            [[sample[feature_name]]], dtype=tf.float32
        )

    for feature_name in CATEGORICAL_FEATURE_NAMES:
        if categorical_is_string[feature_name]:
            value = tf.constant([str(sample[feature_name])])
        else:
            value = tf.constant([sample[feature_name]], dtype=tf.int64)
        sample_inputs[feature_name] = categorical_encoders[feature_name](value)

    return float(model.predict(sample_inputs, verbose=0)[0, 0])


def main():
    keras.utils.set_random_seed(1337)
    dataframe = pd.read_csv(
        "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
    )
    train_ds, val_ds, encoders, categorical_is_string, normalizers = prepare_data(
        dataframe
    )
    model = build_model(encoders, normalizers)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        shuffle=False,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=5,
                restore_best_weights=True,
            )
        ],
        verbose=0,
    )

    metrics = model.evaluate(val_ds, return_dict=True, verbose=0)
    print({name: round(float(value), 4) for name, value in metrics.items()})

    sample = {
        "age": 60,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": "fixed",
    }
    prediction = predict_sample(model, sample, encoders, categorical_is_string)
    print(f"Probabilidade estimada: {prediction:.1%}")
    print(f"Classe prevista: {int(prediction >= 0.5)}")


if __name__ == "__main__":
    main()

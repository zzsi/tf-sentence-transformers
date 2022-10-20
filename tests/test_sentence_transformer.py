import numpy as np
from tensorflow import keras
from tf_sentence_transformers import SentenceTransformer


def test_call():
    layer = SentenceTransformer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", from_pt=False
    )
    inputs = [
        ["This is a test sentence."],
        ["This is another test sentence."],
    ]
    embedding = layer(inputs)
    assert embedding.shape == (2, 384)


def test_train():
    layer = SentenceTransformer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", from_pt=False
    )

    def model_fn(x):
        x = layer(x)
        x = keras.layers.Dense(1)(x)
        return x

    inputs = keras.Input(shape=(None,), dtype="string")
    outputs = model_fn(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001),
        loss=keras.losses.MeanSquaredError(),
    )

    def data_gen():
        batch_size = 2
        while True:
            yield np.array([["this is a sentence example"]] * batch_size), np.array(
                [1] * batch_size, dtype=np.float32
            )

    model.fit(data_gen(), steps_per_epoch=2, epochs=2)


def test_train_jit():
    layer = SentenceTransformer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", from_pt=False, jit_compile=True
    )

    def model_fn(x):
        x = layer(x)
        x = keras.layers.Dense(1)(x)
        return x

    inputs = keras.Input(shape=(None,), dtype="string")
    outputs = model_fn(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001),
        loss=keras.losses.MeanSquaredError(),
    )

    def data_gen():
        batch_size = 2
        while True:
            yield np.array([["this is a sentence example"]] * batch_size), np.array(
                [1] * batch_size, dtype=np.float32
            )

    model.fit(data_gen(), steps_per_epoch=2, epochs=2)

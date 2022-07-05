import os

# this is needed to prevent a cudnn error on some GPU's
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# this reduces the amount of tensorflow logging messages to only errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl  # noqa: E402
import tensorflow as tf  # noqa: E402


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)  # noqa

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],  # noqa
)


class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accurary = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accurary": accurary}


fl.client.start_numpy_client("[::]:8080", client=CifarClient())

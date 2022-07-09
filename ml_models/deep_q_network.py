import keras as keras


class deep_q_model(keras.Model):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(deep_q_model, self).__init__()
        self.dense1 = keras.layers.Dense(fc1_dims, input_shape=(input_dims,), activation='relu')  # noqa: E501
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
        self.dense3 = keras.layers.Dense(n_actions)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)

        return x

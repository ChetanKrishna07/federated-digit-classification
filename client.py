import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import flwr as fl

def gen_random_subset(x_train, y_train, n=5):
  """
  Generates a random subset of the data.
  """

  # Generate random indices without replacement
  random_indices = np.random.choice(len(x_train), size=n, replace=False)

  # Use the random indices to extract elements from the original array
  random_X = x_train[random_indices]
  random_y = y_train[random_indices]

  return random_X, random_y

model = Sequential([
    Conv2D(filters=10,
           kernel_size=3,
           activation='relu',
           input_shape=(28, 28, 1)),
    Conv2D(10, 3, activation='relu'),
    MaxPool2D(2),
    Conv2D(filters=10,
           kernel_size=3,
           activation='relu'),
    Conv2D(10, 3, activation='relu'),
    MaxPool2D(2),
    Flatten(),
    Dense(10, activation='softmax')
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=Adam(),
                metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()
    
    def fit(self, paramenters, config):
        model.set_weights(paramenters)
        
        x_train_loc, y_train_loc = gen_random_subset(x_train, y_train, n=10000)
        model.fit(x_train_loc, y_train_loc, epochs=5, validation_data = (x_test, y_test), batch_size=32)
        return model.get_weights(), len(x_train_loc), {}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return accuracy, len(x_test), {"accuracy": accuracy}
    
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
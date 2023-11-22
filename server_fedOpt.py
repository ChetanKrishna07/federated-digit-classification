import flwr as fl
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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

initial_parameters = fl.common.ndarrays_to_parameters(model.get_weights())

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3), strategy=fl.server.strategy.FedOpt(initial_parameters=initial_parameters))
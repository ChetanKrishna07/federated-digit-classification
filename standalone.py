import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import flwr as fl

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

model.fit(x_train, y_train, epochs=5, validation_data = (x_test, y_test), batch_size=32)
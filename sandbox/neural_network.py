"""
Created by Elias Obreque
Date: 10-12-2023
email: els.obrq@gmail.com
"""


import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from scipy.stats import probplot

rm = 1.738e3


class NeuralNetworkTrainer:
    def __init__(self, file_path, test_size=0.2, batch_size=20, random_state=42):
        self.batch_size = batch_size
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.model = None

    def load_data(self):
        # load pkl file
        data = open(self.file_path, 'rb')
        data_loaded_ = dict(pickle.load(data))
        hist = []
        x_true = np.empty((0, 8))
        beta_true = np.empty((0, 1))
        for key, data_loaded in data_loaded_.items():
            for sim_data in data_loaded:
                historical_state = sim_data['state']
                pos = np.array(historical_state[0]) * 1e-3 / rm # km
                vel = np.array(historical_state[1]) * 1e-3  # km/s
                beta = np.array(historical_state[9])
                x_data = np.hstack([pos, vel, pos**2, vel**2])
                x_true = np.vstack([x_true, x_data])
                beta_true = np.vstack([beta_true, beta[:, 0].reshape(-1, 1)])

        X_train, X_test, y_train, y_test = train_test_split(x_true, beta_true, test_size=self.test_size,
                                                            random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    def load_example(self):
        np.random.seed(self.random_state)
        X = np.linspace(0, 4 * np.pi, 100)
        y_true = np.sin(X)
        noise = np.random.normal(0, 0.1, size=100)
        y_noisy = y_true + noise
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_noisy, test_size=self.test_size, random_state=self.random_state
        )
        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)

        return X_train, X_test, y_train, y_test

    def build_model(self):
        input_shape = 8
        self.model = models.Sequential([
            layers.Dense(10, activation='relu', input_shape=(input_shape,)),
            layers.Dense(10, activation='relu'),
            layers.Dense(10, activation='tanh'),
            layers.Dense(1, activation='sigmoid')
        ])

    def compile_model(self, learning_rate=0.001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self.custom_loss)

    def train_model(self, X_train, y_train, epochs=100, validation_data=None):
        self.model.fit(X_train, y_train, epochs=epochs,
                       steps_per_epoch=50,
                       validation_data=validation_data)
    @staticmethod
    def custom_loss(y_true, y_pred):
        return BinaryCrossentropy(from_logits=True)(y_true, y_pred)

    def save_weights(self, file_path='trained_weights.h5'):
        self.model.save_weights(file_path)

    def load_weights(self, file_path='trained_weights.h5'):
        self.model.load_weights(file_path)

    def plot_results(self, X_train, y_train, X_test, y_test):

        predictions = self.model.predict(X_train)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(y_train, predictions, color='red')
        ax.set_xlabel('X')
        ax.set_ylabel('y')

        predictions = self.model.predict(X_test)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.figure()
        ax.scatter(y_test, predictions, color='red')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        plt.show()


if __name__ == '__main__':
    # Uso de la clase
    folder = "../logs/neutral/fixed/"
    file = "mass_opt_vf_entry_.pkl"
    trainer = NeuralNetworkTrainer(folder + file, batch_size=100)
    X_train, X_test, y_train, y_test = trainer.load_data()

    trainer.build_model()
    trainer.compile_model(learning_rate=0.001)
    # Entrenar el modelo
    trainer.train_model(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    trainer.plot_results(X_train, y_train, X_test, y_test)

    # Guardar los pesos entrenados
    trainer.save_weights()

    # Cargar los pesos entrenados en otra instancia
    new_trainer = NeuralNetworkTrainer('archivo.pkl')
    new_trainer.build_model()
    new_trainer.compile_model(learning_rate=0.001)

    # Cargar los pesos entrenados
    new_trainer.load_weights()

    # Visualizar resultados
    new_trainer.plot_results(X_test, y_test)

import tensorflow_datasets as tfds
import os
import logging
import numpy as np
from tensorflow import keras
from keras import layers
from keras import Model

# initialize logging
logging.basicConfig(format='[%(message)s]', level=logging.INFO)

download_path = "training/data/emnist"

# create mapping from class number to char
digits = [str(i) for i in range(10)]           # 0-9
uppercase = [chr(i) for i in range(65, 91)]    # A-Z  
lowercase = list("abdefghnqrt")                # 11 lowercase letters (that differ from uppercase)
MAPPING = digits + uppercase + lowercase

# Define Test Indices
IDX = 0

class Trainer():

    def __init__(self):
        self.training_images: np.array = None
        self.training_labels: np.array = None
        self.test_images: np.array = None
        self.test_labels: np.array = None
        self.model: Model = None
    
    def load_emnist_dataset(self):
        """
        Download EMNIST Dataset if not eixsting already, unzips it and 
        returns a list of tuples
        """

        # Download Dataset
        logging.info("Downloading emnist")
        path = os.path.join(os.getcwd()) + download_path + "downloads"
        if not os.path.exists(path):

            os.makedirs(download_path, exist_ok=True)
            # Download latest version
            (ds_train, ds_test), ds_info = tfds.load(
                "emnist/balanced",
                split=["train", "test"],
                shuffle_files=True,
                as_supervised=True,
                with_info=True,
                data_dir=download_path
            )

            logging.info(f"emnist dataset saved to {download_path}")

        else:
            logging.info("emnist is already downloaded")

        # convert data to numpy arrays and flatten images (28x28 -> 784)
        ds_train_list = [(image.numpy(), label.numpy()) for image, label in ds_train]
        self.training_images = np.array([t[0].flatten() for t in ds_train_list], dtype=np.float32) / 255.0 # normalize to value between 0 and 1
        self.training_labels = np.array([t[1] for t in ds_train_list])

        ds_test_list = [(image.numpy(), label.numpy()) for image, label in ds_test]
        self.test_images = np.array([t[0].flatten() for t in ds_test_list], dtype=np.float32) / 255.0 # normalize to value between 0 and 1
        self.test_labels = np.array([t[1] for t in ds_test_list]) 
        
    def run_training(self):
        """
        Train Multilayer Perceptron on EMNIST Training Dataset
        """

        # Build simple MLP
        self.model = keras.Sequential([
            layers.Input(shape=(784,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(47, activation="softmax")
        ])

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Trainieren
        self.model.fit(self.training_images, self.training_labels, epochs=10, batch_size=8)

    def make_prediction(self):
        """
        Use trained model to predict outputs from emnist test dataset
        """
        sample = self.test_images[IDX]
        prediction = self.model.predict([sample]) # returns array with probobility for each class
        best_match_idx = np.argmax(prediction[0])
        print("Prediction: ", MAPPING[best_match_idx])
        print("Ground Truth: ", MAPPING[self.test_labels[IDX]])
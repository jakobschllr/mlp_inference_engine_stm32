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

class MLPModel():

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
        
    def run_training(self, save_model=True):
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
        self.model.fit(self.training_images, self.training_labels, epochs=15, batch_size=8)

        if save_model:
            os.makedirs("./models", exist_ok=True)
            self.model.save("./models/model.keras")

    def predict(self):
        """
        Use trained model to predict outputs from emnist test dataset.
        Tries to load model.keras from ./models or uses self.model as fallback, in case the model was just trained using this object.
        """

        try:
            model = keras.models.load_model("./models/model.keras")
        except FileNotFoundError:
            if self.model:
                model = self.model
            else:
                logging.info("No model available. Make sure to train a model before making predictions")
                return


        sample = self.test_images[IDX:IDX+1]  # Keep batch dimension: shape (1, 784)
        prediction = model.predict(sample)  # returns array with probability for each class
        best_match_idx = np.argmax(prediction[0])

        # Test Single Prediction
        print("Prediction: ", MAPPING[best_match_idx])
        print("Ground Truth: ", MAPPING[self.test_labels[IDX]])


        # Calculate Total Accuracy
        predictions = model.predict(self.test_images)
        false_predictions = 0
        for i, pred in enumerate(predictions):
            best_match_idx = np.argmax(pred)
            if MAPPING[best_match_idx] != MAPPING[self.test_labels[i]]:
                false_predictions += 1

        print("Accuracy: ", 1 - (false_predictions / len(predictions)))
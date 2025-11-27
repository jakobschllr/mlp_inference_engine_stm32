import tensorflow_datasets as tfds
import tensorflow
import os
import logging
import numpy as np

# initialize logging
logging.basicConfig(format='[%(message)s]', level=logging.INFO)

download_path = "training/data/emnist"

class Trainer():

    def __init__(self):
        pass

    
    def download_emnist_dataset(self):
        """
        Download EMNIST Dataset if not eixsting already
        """
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

        

        




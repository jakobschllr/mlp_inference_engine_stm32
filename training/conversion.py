import logging
from tensorflow import keras
from keras import Model

# initialize logging
logging.basicConfig(format='[%(message)s]')


class BinaryConverter():
    
    def convert_to_binary(model: Model):

        try:
            model = keras.models.load_model("./models/model.keras")
        except FileNotFoundError:
            logging.info("No model was found for conversion. Make sure to train a model and export it to ./models")
            return

        # convert model data to json
        json_data = model.to_json()

        weights = model.get_weights()

        print(json_data)

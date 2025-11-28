import logging
from training.training import MLPModel
from training.conversion import BinaryConverter

# initialize logging
logging.basicConfig(format='[%(message)s]')

def main(*args):
    model = MLPModel()
    converter = BinaryConverter()

    model.load_emnist_dataset()
    model.run_training()
    model.predict()

    converter.convert_to_binary()


if __name__ == "__main__":
    main()
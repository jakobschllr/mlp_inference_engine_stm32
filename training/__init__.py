import logging
from training import Trainer

# initialize logging
logging.basicConfig(format='[%(message)s]')

def main(*args):
    trainer = Trainer()
    trainer.download_emnist_dataset()


if __name__ == "__main__":
    main()
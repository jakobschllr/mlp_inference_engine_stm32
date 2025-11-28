import logging
from training import Trainer

# initialize logging
logging.basicConfig(format='[%(message)s]')

def main(*args):
    trainer = Trainer()
    trainer.load_emnist_dataset()
    trainer.run_training()
    trainer.make_prediction()



if __name__ == "__main__":
    main()
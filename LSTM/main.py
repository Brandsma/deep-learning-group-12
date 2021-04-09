# !pip install functions
import functions as f
from Text import *
from LSTM_class import *
# !pip install keras
from keras import layers, models, optimizers


def main():
    pass


if __name__ == "__main__":
    if DEBUG:
        log.info("Debug mode is ON")

    filename = "../all.csv"
    main(filename)

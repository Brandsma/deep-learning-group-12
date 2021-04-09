import pandas as pd
import numpy as np
import string
from logger import setup_logger
from tqdm import tqdm

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping

log = setup_logger(__name__)

DEBUG = True
MAX_LEN = 0


def clean_text(input_text):
    text = "".join(
        word for word in input_text if word not in string.punctuation).lower()
    text = text.encode("utf8").decode("ascii", "ignore")
    return text


def load_data(filename):
    df = pd.read_csv(filename)
    return df


def preprocessing(df):
    global MAX_LEN
    log.info("Cleaning dataset...")
    corpus = pd.DataFrame()
    log.info("Corpus has a size of " + str(df.size))
    for idx, elem in tqdm(df.iterrows(), desc="cleaning up: "):
        corpus = corpus.append(
            {"text": clean_text(elem["text"])}, ignore_index=True)
    log.info(
        "Removed all non-utf8 or non-ascii characters, all punctuation and lower-cased it")
    log.info("Found " + str(len(corpus)) + " number of words in the corpus")

    vocab = []
    for line in corpus["text"]:
        words = line.split()
        for word in words:
            vocab.append(word)

    vocab = set(vocab)
    log.info("There are " + str(len(vocab)) + " unique words in the set")

    VOCAB_SIZE = 2000

    # TODO: Maybe limit the number of words we get, instead of just using all the words
    tokenizer = Tokenizer(VOCAB_SIZE)
    tokenizer.fit_on_texts(corpus["text"])
    word2index = tokenizer.word_index
    log.info("Indexed " + str(len(word2index)) +
             " unique words with the tokenizer")
    # print(word2index)

    dictionary = {}
    rev_dictionary = {}
    for word, idx in word2index.items():
        dictionary[word] = idx
        rev_dictionary[idx] = word

    input_sequences = tokenizer.texts_to_sequences(corpus["text"])

    input_data = []
    target = []
    for line in input_sequences:
        for i in range(1, len(line)-1):
            input_data.append(line[:i])
            target.append(line[i+1])

    log.info("Padding...")
    # calc length longest word for padding
    for idx, seq in enumerate(input_data):
        if (len(seq) > 30):
            input_data.pop(idx)
            target.pop(idx)
            continue
        if len(seq) > MAX_LEN:
            MAX_LEN = len(seq)
    log.info("Longest word has " + str(MAX_LEN) + " characters")

    input_data = pad_sequences(
        input_data, maxlen=MAX_LEN, padding="post", truncating="post")
    log.info("Length of all words is " +
             str(len(input_data[0])) + " characters now")

    total_words = VOCAB_SIZE
    target = to_categorical(target, num_classes=total_words)

    log.info("Input data has a shape of " + str(input_data.shape))
    log.info("Target data has a shape of " + str(target.shape))

    return (input_data, target, tokenizer)


def create_LSTM_model(x, y, num_lstm_units=100, drop_rate=0.1, activation_type="softmax", optimizer_type="adam"):
    model = Sequential()
    model.add(Embedding(input_dim=x.shape[0],
                        output_dim=100, input_length=x.shape[1]))
    model.add(LSTM(units=num_lstm_units))
    model.add(Dropout(rate=drop_rate))
    model.add(Dense(units=y.shape[1], activation=activation_type))

    log.info("Compiling...")
    model.compile(loss="categorical_crossentropy", optimizer=optimizer_type)

    if DEBUG:
        model.summary()

    log.info("Created model")

    return model


def create_GRU_model(x, y, num_gru_units=100, drop_rate=0.1):
    model = Sequential()
    model.add(Embedding(input_dim=x.shape[0],
                        output_dim=100, input_length=x.shape[1]))
    model.add(GRU(units=num_gru_units))
    model.add(Dropout(rate=drop_rate))
    model.add(Dense(units=y.shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    if DEBUG:
        model.summary()
    return model


def predict(model, tokenizer, seed_text, num_of_words=10):
    global MAX_LEN
    for _ in range(num_of_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=MAX_LEN, padding="post")
        predicted = np.argmax(model.predict(token_list), axis=-1)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text.title()


def train_model(input_data, target, model, size_batch=32, nr_epochs=3):
    model.fit(input_data, target, batch_size=size_batch,
              epochs=nr_epochs, verbose=1)
    return model


def main(input_file: str, model_type: str = "GRU"):
    log.info("Loading data...")
    df = load_data(input_file)
    log.info("Preprocessing...")
    x, y, tokenizer = preprocessing(df)

    model = None
    if model_type == "LSTM":
        log.info("Creating LSTM model...")
        model = create_LSTM_model(x, y, num_lstm_units=5)
    elif model_type == "GRU":
        log.info("Creating GRU model...")
        model = create_GRU_model(x, y)
    else:
        log.error("Incorrect model type was provided, type " +
                  model_type + " does not exist")

    if model is None:
        return

    log.info("Training model...")
    model = train_model(x, y, model)

    log.info("Predicting...")
    prediction = predict(model, tokenizer, "I have decided that")

    print("Prediction: \n -----")
    print(prediction)
    print("-----")

    log.info(" Done ")


if __name__ == "__main__":
    if DEBUG:
        log.info("Debug mode is ON")

    filename = "../all.csv"
    main(filename)

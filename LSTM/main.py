import functions as f
from Text import *
from LSTM_class import *
from keras import layers, models, optimizers
import pandas as pd


def lstm_model(sequence_length, vocab_size, layer_size, embedding=False):
    model = models.Sequential()
    if embedding:
        model.add(layers.Embedding(vocab_size, layer_size))
        model.add(layers.LSTM(layer_size))
    else:
        model.add(layers.LSTM(layer_size, input_shape=(
            sequence_length, vocab_size)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(vocab_size, activation='softmax'))
    return model


def train(with_embedding):
    max_len = 4
    step = 3

    text_train = Text(input_train)
    text_train.tokens_info()

    seq_train = Sequences(text_train, max_len, step)
    seq_train.sequences_info()

    if with_embedding:
        batch_size_emb = 4096

        params_emb = {
            'sequence_length': max_len,
            'vocab_size': len(text_train),
            'batch_size': batch_size,
            'shuffle': True
        }
        params_emb['embedding'] = True

        train_generator_emb = TextDataGenerator(
            seq_train.sequences, seq_train.next_words, **params_emb)

        model_emb = lstm_model(max_len, len(text_train), 512, embedding=True)

        optimizer = optimizers.Adam(lr=0.005)
        model_emb.compile(loss='categorical_crossentropy', optimizer=optimizer)

        history_emb = model_emb.fit(train_generator_emb,
                                    steps_per_epoch=len(train_generator_emb),
                                    epochs=35,
                                    verbose=1)

        model_emb.save('lstm_model_emb')

        pd.DataFrame.from_dict(history_emb.history).to_csv(
            'data/lstm_loss_history_with_embedding.csv', index=False)
    else:
        #batch_size = 4096
        batch_size = 8

        params = {
            'sequence_length': max_len,
            'vocab_size': len(text_train),
            'batch_size': batch_size,
            'shuffle': True
        }

        train_generator = TextDataGenerator(
            seq_train.sequences, seq_train.next_words, **params)

        model = lstm_model(max_len, len(text_train), 512, embedding=False)
        optimizer = optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        model.fit(train_generator,
                  steps_per_epoch=len(train_generator),
                  epochs=20,
                  verbose=1)

        model.save('lstm_model')

        pd.DataFrame.from_dict(history.history).to_csv(
            'lstm_history.csv', index=False)


def predict(with_embedding, input_prefix=''):
    if with_embedding:
        model_emb = models.load_model('data/out/lstm_model_emb')

        token2ind, ind2token = text_train.token2ind, text_train.ind2token

        text_prefix = Text(input_prefix, token2ind, ind2token)

        pred_emb = ModelPredict(model_emb, text_prefix,
                                token2ind, ind2token, max_len, embedding=True)

        with open("./lstm_with_embedding_output.txt", 'w') as f:
            for idx in range(100):
                print(str(idx + 1) + "/100")
                f.write(pred.generate_sequence(40, temperature=0.7))
                f.write('\n')
    else:
        model = models.load_model('lstm_model')

        token2ind, ind2token = text_train.token2ind, text_train.ind2token

        text_prefix = Text(input_prefix, token2ind, ind2token)

        pred = ModelPredict(model, text_prefix, token2ind, ind2token, max_len)

        with open("./lstm_output.txt", 'w') as f:
            for idx in range(100):
                print(str(idx + 1) + "/100")
                f.write(pred.generate_sequence(40, temperature=0.7))
                f.write('\n')


if __name__ == "__main__":
    if DEBUG:
        log.info("Debug mode is ON")

    with_embedding = True

    main(with_embedding)
    predict(with_embedding)

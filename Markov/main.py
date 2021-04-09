import markovify
import pandas as pd
import re
import html


def load_data(filename):
    df = pd.read_csv(filename)
    return df


def preprocessing(df):
    pass


def train(input_file: str):
    df = load_data(input_file)
    cur_text = ""
    link_pattern = re.compile("http\S+")
    for idx in range(len(df["text"])):
        if df["isRetweet"][idx] == "t":
            continue
        if 'http' in df["text"][idx]:
            sample = re.sub(link_pattern, "", df["text"][idx])
        else:
            sample = df["text"][idx].replace("\n", " ")

        cur_text += sample.replace("\n", " ")
    data_model = markovify.Text(cur_text, state_size=3)
    model_json = data_model.to_json()

    with open("model.json", 'w') as f:
        f.write(model_json)


def predict(word_count: int = 5):
    data_model = None
    with open("model.json", 'r') as f:
        data_model = markovify.Text.from_json(f.read())
    if data_model is None:
        print("Error: data model did not load")

    with open("./markov_output_starting_with_HUGE.txt", "w", encoding="utf-8") as f:
        for idx in range(word_count):
            print(str(idx + 1) + "/" + str(word_count))
            f.write(data_model.make_sentence_with_start(
                "HUGE", strict=False))
            f.write('\n')


if __name__ == "__main__":
    filename = "../all.csv"
    IsTraining = False

    if IsTraining:
        train(filename)

    word_count = 100
    predict(word_count)

# Markov chain very basic model
# Read all tweets, make a transition table
# Which word follows which word, count, and normalize to chance
# Keep in mind when it should stop (140 / 240 characters)
# Make stop also

# Preprocessing
# Tokenize
# Filter undesirable (perhaps regex)
#

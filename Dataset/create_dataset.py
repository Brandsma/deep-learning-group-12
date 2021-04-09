import pandas as pd
import re

def main(path_train):
    # Regex patterns for later
    link_pattern = re.compile("http\S+")
    # Load dataset
    df = pd.read_csv(path_train)

    # shuffle the DataFrame rows
    df = df.sample(frac = 1)

    train_percentage = 0.7
    val_percentage = 0.1

    # Preprocess dataset
    input_train = ""
    input_test = ""
    input_val = ""
    for idx in range(len(df["text"])):
        # Remove all user tags and links
        if 'http' in df["text"][idx]:
            sample = re.sub(link_pattern, "", df["text"][idx])
        else:
            sample = df["text"][idx]

            if (idx / len(df["text"])) <= train_percentage:
                input_train += sample
            elif (idx / len(df["text"])) <= train_percentage + val_percentage:
                input_val += sample
            else:
                input_test += sample

    with open("../train.txt", 'w') as f:
        f.write(input_train)

    with open("../test.txt", 'w') as f:
        f.write(input_test)

    with open("../validation.txt", 'w') as f:
        f.write(input_val)

if __name__=="__main__":
    path_train = "../all.csv"
    main(path_train)

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../all.csv")


trump_tweet_lengths = []
for idx in range(len(df["text"])):
    if df["isRetweet"][idx] == "t":
        continue
    trump_tweet_lengths.append(len(df["text"][idx]))

plt.hist(trump_tweet_lengths)
plt.title("Trump Tweet Length")
plt.xlabel("Character Length")
plt.ylabel("Number of Tweets")
plt.show()

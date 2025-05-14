import pickle
from collections import Counter
import pandas as pd
import re
import random

# Exploration
df = pd.read_csv("archive/True.csv")

print(df.head())
print(df["text"].head())



def training_data(df, test = False):
    # df contains title, date, etc. which is not needed for the ngram model
    # Text also has the location and source, which we dont want for our counter.
    df = df.dropna()
    df = df["text"].apply(lambda x: ' '.join(x.split()[3:]))

    if test:
        return df[0:10]

    return df

def training(df, save=False, n=2):
    ngram_counter = Counter()

    # We dont append each text into one big file, because the last word in a text would be irrelevant to the next one
    # By running the counter on each text seperately we dont muck up the data.
    for entry in df:
        text = entry
        clean_text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        text_list = clean_text.lower().split()

        window = []
        for word in text_list:
            window.append(word)
            print(window)
            if len(window) >= n+1:
                window.pop(0)
                key = tuple(window)
                ngram_counter[key] += 1
            elif len(window) == n:
                key = tuple(window)
                ngram_counter[key] += 1

    if save:
        with open(str(n)+'gram_counter.pickle', 'wb') as handle:
            pickle.dump(ngram_counter, handle)

    return ngram_counter

def weighted_choice(candidates):
    # Chooses the next most common word probabilistically to ensure unique text generation

    total = sum(candidates.values())
    r = random.uniform(0, total)
    upto = 0
    for word, weight in candidates.items():
        upto += weight
        if upto >= r:
            return word
    return random.choice(list(candidates.keys()))

def predict_next_word(text, counter, n=2):
    words = text.split()
    if len(words) < n - 1:
        print("Not enough words to predict", words)
        return None

    context = tuple(words[-(n - 1):])  # last n-1 words

    # Find all n-grams with this prefix
    candidates = {ngram[-1]: count for ngram, count in counter.items() if ngram[:-1] == context}

    if not candidates:
        print("No candidates found", context)
        return None

    return weighted_choice(candidates)

def predict_text(text = "", n=2, counter=Counter(), length = 30):
    words = text.split()
    if len(words) < length:
        most_common = predict_next_word(text, counter, n=n)
        if most_common is None:
            print("Prediction ended early")
            return text
        text += " " + most_common
        return predict_text(text, n=n, counter=counter, length=length)
    else:
        return text

def load_model(n):
    with open(str(n)+'gram_counter.pickle', 'rb') as handle:
        if handle:
            return pickle.load(handle)
        else:
            print("No model found")


# Training

n = 5
df = training_data(df)
ngram = training(df, save=True, n=n)

# Predicting

ngram = load_model(n)

start = "donald trump has said"

prediciton = predict_text(start, n, ngram)

print(f"------------------\n\nPREDICTION COMPLETE\n\nInput: {start}\n\nOutput: {prediciton}")
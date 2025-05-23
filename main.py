import pickle
from collections import Counter
import pandas as pd
import re
import random
import spacy
import os







# Initialising ngram models
def load_model(pickle_name):
    try:
        with open(pickle_name, 'rb') as handle:
            return pickle.load(handle)
    except FileNotFoundError:
        print(f"{n} gram model not found")
        return None

ngram_models = {}

pattern = re.compile(r'(\d+)gram_counter(?:_spacy)?\.pickle')

folder_path = "./archive/models"

for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        n = int(match.group(1))
        model = load_model(os.path.join(folder_path, filename))
        if model:
            print(f"{filename} loaded as {n} gram model")
            ngram_models[n] = model

nlp = spacy.load("en_core_web_sm")


def custom_tokenize(text):

    # Split common contractions
    print(text)
    doc = nlp(text)
    print(doc)
    tokens = [token.text for token in doc]

    return tokens



def training_data(df, test = False):
    # df contains title, date, etc. which is not needed for the ngram model
    # Text also has the location and source, which we dont want for our counter.
    df = df.dropna()
    df = df["text"].apply(lambda x: ' '.join(x.split()[3:]))

    if test:
        return df[0:10]

    return df

def training(df, save=False, n=2, custom_id = ""):
    ngram_counter = Counter()

    # We dont append each text into one big file, because the last word in a text would be irrelevant to the next one
    # By running the counter on each text seperately we dont muck up the data.
    for entry in df:
        text = entry
        text_list = custom_tokenize(text)

        window = []
        for word in text_list:
            window.append(word)

            if len(window) >= n+1:
                window.pop(0)
                key = tuple(window)
                ngram_counter[key] += 1
            elif len(window) == n:
                key = tuple(window)
                ngram_counter[key] += 1

    if save:
        with open(folder_path + str(n)+'gram_counter' + custom_id + '.pickle', 'wb') as handle:
            pickle.dump(ngram_counter, handle)

    return ngram_counter

def weighted_choice(candidates, temperature = 1):
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

    # replaced by interpolated_next_word
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

def interpolated_next_word(text, n):
    words = text.split()

    candidates = {}

    weights = {
        2: 0.001,
        3: 0.029,
        4: 0.27,
        5: 0.7,
    }

    for ngram in range(1, n+1):
        context = tuple(words[-(ngram - 1):])  # last n-1 words

        model = ngram_models.get(ngram, None)

        if model is None:
            continue

        weight = weights.get(ngram)

        new_candidates = {ngram[-1]: count*weight for ngram, count in ngram_models[ngram].items() if ngram[:-1] == context}

        candidates.update(new_candidates)

    return weighted_choice(candidates)

def predict_text(text = "", n=2, length = 30):
    words = text.split()
    if len(words) < length:
        most_common = interpolated_next_word(text, n=n)
        if most_common is None:
            print("Prediction ended early")
            return text
        text += " " + most_common
        return predict_text(text, n=n, length=length)
    else:
        return text

def train_model(n=5, custom_id = ""):
    # Exploration
    df = pd.read_csv("archive/True.csv")

    print(df.head())
    print(df["text"].head())

    df = training_data(df)
    ngram = training(df, save=True, n=n, custom_id = custom_id)
    print(f"{n} gram model trained")

for n in range(1,3):
    train_model(n, "_spacy")

while True:
    text = input("Enter a text: ")
    prediction = predict_text(text)
    print("Model predicted: ", prediction)
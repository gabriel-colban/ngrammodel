import spacy
import pandas as pd
import pickle

nlp = spacy.load("en_core_web_sm")

def training_data(df, test = False):
    # df contains title, date, etc. which is not needed for the ngram model
    # Text also has the location and source, which we dont want for our counter.
    df = df.dropna()
    df = df["text"].apply(lambda x: ' '.join(x.split()[3:]))

    if test:
        return df[0:10]

    return list(df)

df = pd.read_csv("archive/True.csv")

training_list = training_data(df)

docs = nlp.pipe(training_list, batch_size=50, disable=['parser', 'tagger', "ner"])

token_lists = []

count = 0

for doc in docs:
    tokens = [
        "<NUM>" if token.like_num else token.text.lower()
        for token in doc
        if not token.is_punct and not token.is_space
    ]

    count += 1

    token_lists.append(tokens)

    if count % 250 == 0:
        print(f"Processed {count} / {len(training_list)} texts")

print("tokenization done \n\n", token_lists[0:5])

with open('training_token_lists.pickle', 'wb') as handle:
    pickle.dump(token_lists, handle)


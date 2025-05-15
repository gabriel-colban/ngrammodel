# ngrammodel
Learning and expanding upon ngram models, using training data from kaggle. (News stories)

Note: Due to GitHub limitations, model and training data are not uploaded yet. I plan to use GitHub Large File Storage (LFS) or provide links to external sources later.


# Todo
✔️ - Done
❌ - To be done
👨‍💻 - Working on

### [✔️] n-gram model frame

I've built a rudamentary n-gram model, and trained on data from [Kaggle.](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

It will automatically pickle as ngram_counter.pickle, where n will be the number youve selected for your context depth. It also includes a function for loading, and a very simple text generator that includes probabilistic next-word choice. 

### [❌] Temperature

### [👨‍💻] Interpolating between low to high n-models

Ive added interpolation using 2-5 gram models. 

It will now add the probabilities of the next word from each model, with a weight assigned each of them.

Initially i set the weights as such:

2gram - 0.1
3gram - 0.2
4gram - 0.3
5gram - 0.4

But this resultet in incoherent text, i suspect this is due to the size of the corpus. Lower N models have so many ngrams to choose between that they have a bigger probability of being chosen despite the lower weights.

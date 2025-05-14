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
### [❌] Lower N Fallback
### [❌] Interpolating between low to high n-models

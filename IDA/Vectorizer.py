from gensim import corpora,models

from csv import DictReader, DictWriter

import logging

def tokenize(text):
    texts = [word for word in text.lower().split()]
    return texts
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    train = list(DictReader(open("../data/data_train.csv", 'r')))
    test = list(DictReader(open("../data/data_test.csv", 'r')))
    texts=[]
    for x in train:
        texts.append(tokenize(x['title']))
    dictionary = corpora.Dictionary(texts)
    #print(dictionary.token2id)
    dictionary.save('dictionary.dict')

    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpora.MmCorpus.serialize('train.mm', tfidf[corpus])

    texts = []
    for x in test:
        texts.append(tokenize(x['title']))
    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpora.MmCorpus.serialize('test.mm', tfidf[corpus])

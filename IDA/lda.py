import gensim,logging
from gensim import corpora
import numpy as np
from numpy import array
import logging
from csv import DictReader, DictWriter
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression
def train_model():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    dictionary=corpora.Dictionary.load('dictionary.dict')
    mm = gensim.corpora.MmCorpus('train.mm')
    lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=200, update_every=1, chunksize=2000,
                                          passes=2 )
    lda.save("reuters.lda")
def tokenize(text):
    texts = [word for word in text.lower().split()]
    return texts
def getTopicForQuery (question,lda):

    topic_vec = lda[question]

    max=0
    index=0
    for i in xrange(len(topic_vec)):
        if topic_vec[i][1]>max:
            max=topic_vec[i][1]
            index = i
    return topic_vec[index][0]
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    '''rain_model()
    dictionary = corpora.Dictionary.load('dictionary.dict')
    lda=gensim.models.ldamodel.LdaModel.load('reuters.lda')
    print lda
    train_topics=[]
    mm = gensim.corpora.MmCorpus('train.mm')
    curidx=0
    for x in mm:
        if(len(x)>0):
            curidx += 1
            train_topics.append([getTopicForQuery(x, lda)])
        print curidx
    print train_topics

    curidx=0
    test_topics=[]
    mm = gensim.corpora.MmCorpus('test.mm')
    for x in mm:
        if len(x)>0:
            curidx += 1
            test_topics.append([getTopicForQuery(x, lda)])
        print curidx

    print len(train_topics)
    print len(test_topics)

    f=open('train_topics.txt','w')
    for ii in train_topics:
        f.write(str(ii)+'\n')
    f.close()
    f=open('test_topics.txt','w')
    for ii in test_topics:
        f.write(str(ii)+'\n')
    f.close()'''

    f=open("train_topics.txt")
    x_train=[]
    while 1:
        line = f.readline()
        if not line:
            break
        x_train.append([int(line[1:-2])])

    f=open("test_topics.txt")
    x_test=[]
    while 1:
        line = f.readline()
        if not line:
            break
        x_test.append([int(line[1:-2])])

    labelCol = 'topic'

    train = list(DictReader(open("../data/data_train.csv", 'r')))
    test = list(DictReader(open("../data/data_test.csv", 'r')))
    labels = []
    for line in train:
        if not line[labelCol] in labels:
            labels.append(line[labelCol])

    for line in test:
        if not line[labelCol] in labels:
            labels.append(line[labelCol])


    y_train=[]
    for x in train:
        if len(x['title'])>0:
            y_train.append(labels.index(x[labelCol]))
    #y_train = array(list(labels.index(x[labelCol]) for x in train))
    print len(y_train)
    y_test=[]
    for x in test:
        if len(x['title'])>0:
            y_test.append(labels.index(x[labelCol]))
    #y_train = array(list(labels.index(x[labelCol]) for x in train))
    print len(y_test)
    #print(len(train), len(y_train))
    #print(set(y_train))

    # Train classifier
    lr = OneVsRestClassifier(LinearSVC(),n_jobs=-1)
    lr.fit(x_train, y_train)
    print 'training acc:',
    print lr.score(x_train,y_train)
    print 'test acc:',
    print lr.score(x_test,y_test)
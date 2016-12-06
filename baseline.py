from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
import random
from pattern.en import singularize, lemma, parse

from scipy.sparse import csr_matrix
from collections import defaultdict

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'

class Featurizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.flag = True
        #self.vectorizer = CountVectorizer()

    def convert(self, word):

        before = word 
        word = lemma(word)  # Base form, e.g., are => be.
        word = singularize(word) # singular

        #print before, word

        return word

    def normalize(self, examples):
        for sentence in examples :
                t = parse(sentence,tags=False,chunks=False)
                news = ''
                for row in t.split():
                        for word in row:
                                news = news + self.convert(word[0]) + ' '
                if self.flag == True:
                        self.flag = False
                        #print type(sentence), sentence #debug
                        #print type(news), news
                yield(news)

    def train_feature(self, examples):
        #examples = self.normalize(examples)
        #for it in examples:
        #        print it
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        #examples = self.normalize(examples)
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

def validater(x_train, y_train):
        cut = int(x_train.shape[0] * 0.8)
        index = np.arange(x_train.shape[0])
        bestScore = 0

        for i in xrange(5):
                random.shuffle(index)
                x = x_train[index[:cut], :]
                y = y_train[index[:cut]]

                lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
                lr.fit(x,y)

                score = lr.score(x_train[index[cut:], :], y_train[index[cut:]])
                if bestScore < score :
                        bestScore = score
                        ret = lr
        return ret, bestScore

def baselineAcc(x,y,contribute):
        rows,cols = x.nonzero()
        result = [defaultdict(lambda: 0.0) for i in xrange(x.shape[0])]
        best = defaultdict(lambda:0.0)
        resultLabel = [None] * x.shape[0]

        print 'contribute calculation begin!'
        counting = 0
        for r,c in zip(rows,cols):
                counting += 1 
                for (label,rate) in contribute[c].iteritems():
                        result[r][label] += rate
                        if result[r][label] > best[r] :
                                best[r] = result[r][label]
                                resultLabel[r] = label
        print 'contribute calculation end!'
        cnt = 0
        for r in xrange( x.shape[0] ):
                if resultLabel[r] == y[r] :
                        cnt = cnt + 1

        res = cnt * 1.0 / x.shape[0] 
        print res 
        return res 
                
def baseline(train_x,train_y,test_x,test_y):
        rows,cols = train_x.nonzero()
        contribute = [defaultdict(lambda:0.0) for i in xrange(train_x.shape[1])]# * train_x.shape[1]
        for r,c in zip(rows,cols):
            contribute[c][train_y[r]] += train_x[r,c]
            #print "row = %d, column = %d, value = %s" % (r,c,train_x[r,c])
        baselineAcc(train_x,train_y,contribute)
        baselineAcc(test_x,test_y,contribute)

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("./data/data.csv", 'r')))
    #test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    feat = Featurizer()

    labelCol = 'topic'

    labels = []
    for line in train:
        if not line[labelCol] in labels:
            labels.append(line[labelCol])
    cut = len(train) * 4 / 5 
    test = train[cut:]
    train = train[:cut]

    print("Label set: %s" % str(labels))
    print len(labels)
    x_train = feat.train_feature(x['body'] for x in train)
    y_train = array(list(labels.index(x[labelCol]) for x in train))       
    x_test = feat.test_feature(x['body'] for x in test)
    y_test = array(list(labels.index(x[labelCol]) for x in test) )
    baseline(x_train,y_train,x_test,y_test)

    #x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)
    
    #for i in xrange(len(y_train)):
    #    if y_train[i]==True:
    #            print train[i][kTEXT_FIELD]

    #print(len(train), len(y_train))
    #print(set(y_train))

    # Train classifier
    #lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    #lr.fit(x_train, y_train)
    #lr, acc = validater(x_train, y_train)

    #feat.show_top10(lr, labels)
    #print('The training accuracy rate:')
    #print lr.score(x_train,y_train)
    #print('The testing accuracy rate:')
    #print acc

    #predictions = lr.predict(x_test)
    #o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
    #o.writeheader()
    #for ii, pp in zip([x['id'] for x in test], predictions):
    #    d = {'id': ii, 'spoiler': labels[pp]}
    #    o.writerow(d)
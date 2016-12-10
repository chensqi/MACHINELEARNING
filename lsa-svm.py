import random
import numpy as np
from numpy import array
from csv import DictReader, DictWriter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from pattern.en import singularize, lemma, parse
from scipy.sparse import csr_matrix
from collections import defaultdict
from scipy.sparse.linalg import svds
from scipy.sparse import diags

def preProcess(s):
    ii=0
    while(1):
        if(ii>=len(s)):
            break;
        if (not ((s[ii] >= 'a' and s[ii] <= 'z')or (s[ii] >= '0' and s[ii] <= '9') or s[ii]==' ')):
            if(s[ii]>='A'and s[ii]<= 'Z'):
                s = s[0:ii] + ' ' + s[ii:]
                ii = ii + 1
            else:
                s = s[0:ii] + ' ' + s[ii]+' '+s[ii+1:]
                ii = ii+2
        ii+=1
    return s.lower()
class Featurizer:
    def __init__(self):
        #doc=list(DictReader(open("../data/spoilers/words.csv", 'r')))
        #stops=[]
        #for x in doc:
        #    stops.append(x['words'])
        #self.vectorizer = CountVectorizer()
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)

    def train_feature(self, examples):
        fit = self.vectorizer.fit_transform(examples)
        length=[]
        for x in fit:
            length.append([x.count_nonzero()/200.0])
        #fit=scipy.sparse.csr_matrix(scipy.sparse.hstack([fit,length]))
        #print fit[2]
        return fit

    def test_feature(self, examples):
        fit = self.vectorizer.transform(examples)
        length=[]
        for x in fit:
            length.append([x.count_nonzero() / 200.0])
        #fit=scipy.sparse.csr_matrix(scipy.sparse.hstack([fit,length]))
        return fit
if __name__ == "__main__":

    train = list(DictReader(open("./data/data_train.csv", 'r')))
    test = list(DictReader(open("./data/data_test.csv", 'r')))

    feat = Featurizer()

    labelCol = 'topic'

    labels = []
    for line in train:
        if not line[labelCol] in labels:
            labels.append(line[labelCol])

    for line in test:
        if not line[labelCol] in labels:
            labels.append(line[labelCol])

            # print("Label set: %s" % str(labels))
            # print len(labels)
    x_train = feat.train_feature(x['title'] + ' ' + x['body'] for x in train)
    y_train = array(list(labels.index(x[labelCol]) for x in train))
    x_test = feat.test_feature(x['title'] + ' ' + x['body'] for x in test)
    y_test = array(list(labels.index(x[labelCol]) for x in test))

    control = 1000 
    U, S, VT = svds(csr_matrix.transpose(x_train),k=control)

    cut = control 
    U = U[:,:cut]
    S = diags( S[:cut] , 0)
    VT = VT[:cut,:]

    #x_train = x_train * U
    x_train = csr_matrix.transpose( csr_matrix(S*VT))
    x_test = x_test * U
    #print(set(y_train))

    # Train classifier
    lr = OneVsRestClassifier(LinearSVC(random_state=0),n_jobs=-1)
    lr.fit(x_train, y_train)
    print 'training acc:',
    print lr.score(x_train,y_train)
    print 'test acc:',
    print lr.score(x_test,y_test);

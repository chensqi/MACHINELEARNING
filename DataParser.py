from sgmllib import SGMLParser
from csv import DictReader, DictWriter
class dataParser(SGMLParser):
    def __init__(self):
        SGMLParser.reset(self)
        self.title=[]
        self.topic=[]
        self.body=[]
        self.labels=[]
        self.curti=''
        self.curtop=''
        self.curbody=''
        self.label=0
        self.istopic=False
        self.istitle=False
        self.isbody=False
        self.isd=False
        self.islabel=False
        self.hastopic=False
        self.otrain = DictWriter(open("data/data_train.csv", 'w'), [u"topic", u"title",u"body"])
        self.otrain.writeheader()
        self.otest = DictWriter(open("data/data_test.csv", 'w'), [u"topic", u"title",u"body"])
        self.otest.writeheader()
    def readData(self,path):
        f=open(path,'r')
        SGMLParser.feed(self,f.read())
    def start_reuters(self,attrs):
        for i,j in attrs:
            if i=='topics':
                if j=='YES':
                    self.hastopic=True
                elif j=='NO':
                    self.hastopic=False
            elif i=="lewissplit":
                if j=='TRAIN':
                    self.label=0
                elif j=='TEST':
                    self.label=1
                else:
                    self.label=2

    def end_reuters(self):
        if(not len(self.curtop)==0):
            self.topic.append(self.curtop)
            self.body.append(self.curbody)
            self.title.append(self.curti)
            self.labels.append(self.label)
            self.curti = ''
            self.curtop = ''
            self.curbody = ''

    def start_topics(self,attrs):
        self.istopic=True
    def end_topics(self):
        if self.istopic:
            self.istopic=False

    def start_title(self,attrs):
        self.istitle=True
    def end_title(self):
        if self.istitle:
            self.istitle=False

    def start_d(self,attrs):
        self.isd=True
    def end_d(self):
        if self.isd:
            self.isd=False

    def start_body(self,attrs):
        self.isbody=True
    def end_body(self):
        if self.isbody:
            self.isbody=False
    def handle_data(self, data):
        if self.hastopic:
            if self.isd:
                if self.istopic:
                    if(len(self.curtop)==0):
                        #self.curtop+='/'
                        self.curtop+=data
            elif self.istopic:
                if (len(self.curtop) == 0):
                    #self.curtop += '/'
                    self.curtop += data

            elif self.isbody:
                self.curbody+=data

            elif self.istitle:
                self.curti+=data
    def printTopic(self):
        print len(self.topic)
        print len(self.body)
        print self.topic
    def writeData(self):
        for ii, jj,kk,ll in zip(self.topic, self.title,self.body,self.labels):
            # print ii, jj, kk
            if(ll==0):
                d = {'topic': ii.decode("utf-8", 'ignore'), 'title': jj.decode("utf-8", 'ignore'),
                     'body': kk.decode("utf-8", 'ignore')}
                self.otrain.writerow(d)
            elif(ll==1):
                d = {'topic': ii.decode("utf-8", 'ignore'), 'title': jj.decode("utf-8", 'ignore'),
                     'body': kk.decode("utf-8", 'ignore')}
                self.otest.writerow(d)


if __name__ == "__main__":
    d=dataParser()
    path='data/reuters21578/reut2-0'
    for x in xrange(0,22):
        print x
        npath=path+"%02d"% (x)+".sgm"
        d.readData(npath)
    d.writeData()



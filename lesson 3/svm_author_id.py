#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###


print ("This is my code.\n")


features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

from sklearn import svm

## clf = svm.SVC(kernel="linear")
clf = svm.SVC(kernel="rbf", C=10000.0)

print ("C = ", getattr(clf, 'C'))


t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
print (clf.score(features_test, labels_test))
print "scoring time:", round(time()-t1, 3), "s"

pred = clf.predict(features_test)
pred.sum()  ## number of Chris's emails


'''
linear kernel:
with entire data set:

training time: 163.169 s
0.984072810011
scoring time: 17.188 s


with smaller data set:

training time: 0.089 s
0.884527872582
scoring time: 0.961 s

rbf kernel:

training time: 0.101 s
0.616040955631
scoring time: 1.101 s

C=10.0

training time: 0.102 s
0.616040955631
scoring time: 1.101 s

C=100.0

training time: 0.101 s
0.616040955631
scoring time: 1.101 s

C=1000.0


training time: 0.097 s
0.821387940842
scoring time: 1.052 s

C=10000.0

training time: 0.096 s
0.892491467577
scoring time: 0.886 s

('C = ', 10000.0)
training time: 0.099 s
0.892491467577
scoring time: 0.885 s

for x in dir(clf):
  print (x, getattr(clf, x))

complete dataset:

('C = ', 10000.0)
training time: 109.979 s
0.990898748578
scoring time: 11.163 s

'''



#########################################################


# Load libraries
from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn import preprocessing

#Read train data
file1 = open("/Users/poojithaamin/Desktop/DOCS/SJSU/SEM3/255/Assignment2/data/train.dat")
lines = file1.readlines()

#Read train labels
file2 = open("/Users/poojithaamin/Desktop/DOCS/SJSU/SEM3/255/Assignment2/data/train.labels")
lines2 = file2.readlines()

#Read test data
file3 = open("/Users/poojithaamin/Desktop/DOCS/SJSU/SEM3/255/Assignment2/data/test.dat")
lines3 = file3.readlines()

import numpy
X = numpy.loadtxt(lines)
#X = StandardScaler().fit(X)
#X = preprocessing.normalize(X, norm='l2')

print(X[1,0])
print(X.shape)

Y = numpy.loadtxt(lines2)
print(Y[1])
print(Y.shape)

X_CLP = numpy.loadtxt(lines3)


#############
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from collections import Counter

#Print distribution of classes
print(sorted(Counter(Y).items()))
'''
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X, Y)
print(sorted(Counter(y_resampled).items()))
'''

#Split the data into train and validation set
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.20, random_state=2)

#To handle imbalanced data
#X_train, Y_train = SMOTE(random_state=42, k_neighbors=3).fit_sample(X_train, Y_train)

pipe = Pipeline([  
                                ('dimred',SelectKBest(f_classif, k=60)),
                      ('clf', ExtraTreesClassifier(n_estimators=400, max_depth=50, min_samples_split=5, random_state=2)),
])

clfs = []
#clfs.append(RandomForestClassifier(n_estimators=200, max_depth=50, random_state=42, criterion='entropy'))
clfs.append(ExtraTreesClassifier(n_estimators=200, min_samples_split=6, random_state=1, max_features=None, class_weight='balanced_subsample'))

for classifier in clfs:
    pipe.set_params(clf = classifier)
    pipe = pipe.fit(X_train, Y_train)

    print(str(classifier))
    y_pred = pipe.predict(X_validation)

    print(f1_score(y_pred, Y_validation, average="macro"))
    print(precision_score(y_pred, Y_validation, average="macro"))
    print(recall_score(y_pred, Y_validation, average="macro")) 


pipe = pipe.fit(X, Y)

y_CLP = pipe.predict(X_CLP)
y_CLP = [int(round(value,0)) for value in y_CLP]
print(len(y_CLP))
print(y_CLP[5295])

#y_CLP.tofile('/Users/poojithaamin/Desktop/DOCS/SJSU/SEM3/255/Assignment2/format.dat', sep="\n", format="%s")

#Convert float values to integer
thefile = open('/Users/poojithaamin/Desktop/DOCS/SJSU/SEM3/255/Assignment2/format.dat', 'w+')
for item in y_CLP:
  thefile.write("%s\n" % item)
thefile.close()  
#############

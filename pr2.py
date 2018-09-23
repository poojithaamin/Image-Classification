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

# Load libraries
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

file1 = open("/Users/poojithaamin/Desktop/DOCS/SJSU/SEM3/255/Assignment2/data/train.dat")
lines = file1.readlines()

#Read test data
file2 = open("/Users/poojithaamin/Desktop/DOCS/SJSU/SEM3/255/Assignment2/data/train.labels")
lines2 = file2.readlines()


file3 = open("/Users/poojithaamin/Desktop/DOCS/SJSU/SEM3/255/Assignment2/data/test.dat")
lines3 = file3.readlines()

import numpy
X = numpy.loadtxt(lines)
#X = StandardScaler().fit(X)
#X = preprocessing.normalize(X, norm='l2')

print(X[1])
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

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=2)
                      
text_mnb_stemmed = Pipeline([   #('minmax', preprocessing.FunctionTransformer(np.log1p)),
                                ('cluster', cluster.FeatureAgglomeration(n_clusters=800)),
                        #('pca', PCA(n_components=20, svd_solver='randomized', random_state=2)),
                      ('clf', KNeighborsClassifier(n_neighbors=3, weights='uniform')),
])

clfs = []
clfs.append(KNeighborsClassifier(n_neighbors=3, weights='uniform'))

for classifier in clfs:
    text_mnb_stemmed.set_params(clf = classifier)
    text_mnb_stemmed = text_mnb_stemmed.fit(X_train, Y_train)

    print(str(classifier))
    y_pred = text_mnb_stemmed.predict(X_validation)
    print(f1_score(y_pred, Y_validation, average="macro"))
    print(precision_score(y_pred, Y_validation, average="macro"))
    print(recall_score(y_pred, Y_validation, average="macro")) 

y_CLP = text_mnb_stemmed.predict(X_CLP)
y_CLP.tofile('/Users/poojithaamin/Desktop/DOCS/SJSU/SEM3/255/Assignment2/format.dat', sep="\n", format="%s")

#############

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
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN

#Print distribution of classes
print(sorted(Counter(Y).items()))
'''
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X, Y)
print(sorted(Counter(y_resampled).items()))
'''

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.20, random_state=2)

#X_train, Y_train = SMOTE(random_state=42, k_neighbors=3).fit_sample(X_train, Y_train)

text_mnb_stemmed = Pipeline([  
                                ('dimred',SelectKBest(f_classif, k=60)),
                      ('clf', ExtraTreesClassifier(n_estimators=400, max_depth=50, min_samples_split=5, random_state=2)),
])

clfs = []
#clfs.append(RandomForestClassifier(n_estimators=200, max_depth=50, random_state=42, criterion='entropy'))
clfs.append(ExtraTreesClassifier(n_estimators=220, min_samples_split=6, random_state=1, max_features=None, class_weight='balanced_subsample'))

for classifier in clfs:
    text_mnb_stemmed.set_params(clf = classifier)
    text_mnb_stemmed = text_mnb_stemmed.fit(X_train, Y_train)

    print(str(classifier))
    y_pred = text_mnb_stemmed.predict(X_validation)

    print(f1_score(y_pred, Y_validation, average="macro"))
    print(precision_score(y_pred, Y_validation, average="macro"))
    print(recall_score(y_pred, Y_validation, average="macro")) 


text_mnb_stemmed = text_mnb_stemmed.fit(X, Y)

y_CLP = text_mnb_stemmed.predict(X_CLP)
y_CLP = [int(round(value,0)) for value in y_CLP]
print(len(y_CLP))
print(y_CLP[1])

#y_CLP.tofile('/Users/poojithaamin/Desktop/DOCS/SJSU/SEM3/255/Assignment2/format.dat', sep="\n", format="%s")

thefile = open('/Users/poojithaamin/Desktop/DOCS/SJSU/SEM3/255/Assignment2/format.dat', 'w')
for item in y_CLP:
  thefile.write("%s\n" % item)
  
#############

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
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.20, random_state=2)
                      
rfc = RandomForestClassifier()
rfe_model = RFE(rfc, 4, step=1)
rfe_model = rfe_model.fit(X_train, Y_train)

# evaluate the model on testing set
pred_y = rfe_model.predict(test_x)
predictions = [round(value) for value in pred_y]

print(f1_score(predictions, Y_validation, average="macro"))
print(precision_score(predictions, Y_validation, average="macro"))
print(recall_score(predictions, Y_validation, average="macro")) 


y_CLP = text_mnb_stemmed.predict(X_CLP)
y_CLP.tofile('/Users/poojithaamin/Desktop/DOCS/SJSU/SEM3/255/Assignment2/format.dat', sep="\n", format="%s")

#############


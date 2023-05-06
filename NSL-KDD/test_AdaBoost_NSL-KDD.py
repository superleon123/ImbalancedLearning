# -*- coding: utf-8 -*-
"""
Created on Fri May  5 22:28:38 2023

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:12:36 2022
using Random undersampling on majority class
balanced on trainning data
@author: Administrator
"""

from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss, classification_report, ConfusionMatrixDisplay
# , \     plot_confusion_matrix
    
from collections import Counter
from sklearn.metrics import accuracy_score
import sys
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
# import xgboost as xgb
import multiprocessing
import joblib
from sklearn.ensemble import AdaBoostClassifier #as adb
from imblearn.metrics import classification_report_imbalanced
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwangi

#traing dataset
train_x = pd.read_csv('KDDTrain_x.csv',header=None)
train_Y = pd.read_csv('KDDTrain_y.csv',header=None)
train_x.drop(train_x.columns[[115,116,117]], axis=1, inplace=True)  #drop unused columns
train_Y[0]=list(train_Y[0].map({'benign':0, 'dos':1, 'probe':2, 'r2l':3, 'u2r':4}))

print("train_x", train_x.head())
print("train_y", train_Y.head())

#testing dataset
test_x = pd.read_csv('KDDTest_x.csv',header=None)
test_Y = pd.read_csv('KDDTest_y.csv',header=None)
test_x.drop(test_x.columns[[115,116,117]], axis=1, inplace=True)
test_Y[0]=list(test_Y[0].map({'benign':0, 'dos':1, 'probe':2, 'r2l':3, 'u2r':4}))
print("test_x", test_x.head())
print("test_y", test_Y.head())

# train_Y = train_Y.values.ravel()
# test_Y  = test_Y.values.ravel()
# print('Original dataset shape %s' % Counter(train_Y))

train_Y = train_Y.values.ravel()
test_Y  = test_Y.values.ravel()
print('Original train dataset shape %s' % Counter(train_Y))
print('Original test dataset shape %s' % Counter(test_Y))

######--------------- RandomUnderSampler-------------------------
# mean_class_size = int(pd.Series(train_Y).value_counts().sum()/5)
# print("mean_class_size", mean_class_size)

# # reduce class 0 and 1 to mean class size
# ratio= {0: mean_class_size,
#      1: mean_class_size
#      #'probe': mean_class_size,
#      #'r2l': mean_class_size,
#      #'u2r': mean_class_size
#      }

# # check if ratio para is still used
# # print("before \n", pd.Series(train_Y).value_counts())
# # print('Original dataset shape %s' % Counter(train_Y))

# #from imblearn.under_sampling import RandomUnderSampler
# rus = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
# # rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
# train_x, train_Y = rus.fit_resample(train_x, train_Y)
# print('now dataset shape %s' % Counter(train_Y))


# xgb_model = xgb.XGBClassifier()

adb_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
# clf = GridSearchCV(adb_model, {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}, verbose=1, n_jobs=2)


# param_grid = [
#     {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}, 
#     {'objective': ['multi:softmax'], 'verbose': [1], 'n_jobs':[-1]},
#     ]
    
param_grid = {'base_estimator__max_depth': [2, 4, 6],       #[i for i in range(2,11,2)]
        'base_estimator__min_samples_leaf':[5,10],
        'n_estimators':[50, 100, 200],
        'learning_rate':[0.01, 0.1]
        } 
        
    

# clf = GridSearchCV(xgb_model, {'max_depth': [2, 4, 6],
#                                'n_estimators': [50, 100, 200]}, verbose=1,
#                                 n_jobs=-1)
# clf = GridSearchCV(xgb_model, param_grid, cv=2)

# clf = GridSearchCV(xgb_model, param_grid, cv=5)
clf = GridSearchCV(adb_model, param_grid, cv=5, n_jobs=-1)

clf.fit(train_x, train_Y)

# save model
# clf.best_estimator_.save_model('myRUSBoost.json')
joblib.dump(clf, "myRUSBoost.pkl")

print(clf.best_score_)
print(clf.best_params_)


#evaluation with no sampling on test data
#pred_y = xgb_model.predict(x_test)
pred_y = clf.predict(test_x)
results = confusion_matrix(test_Y, pred_y)
error = zero_one_loss(test_Y, pred_y)
target_names = ['benign 0', 'dos 1', 'probe 2', 'r2l 3', 'u2r 4']

print("test result: \n" , results)
print("test error:", error)
# print("validation:", accuracy_score(test_Y, pred_y))
print("test validation:", accuracy_score(test_Y, pred_y))
print(classification_report(test_Y, pred_y, target_names=target_names))
print(classification_report_imbalanced(test_Y, pred_y, digits=4,target_names=target_names))

cm = confusion_matrix(test_Y, pred_y, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
_ = disp.ax_.set_title("AdaBoost")
plt.show()






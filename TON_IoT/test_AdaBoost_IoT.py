# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:48:10 2023

@author: yishulou
"""

#  -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 11:26:50 2022

@author: Administrator
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:12:36 2022
Ton_IoT dataset
using ADASYN and tomek link, and xgboost
with 20% dataset
balanced data model on test data
@author: Administrator
"""

# from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss, classification_report, ConfusionMatrixDisplay
from collections import Counter
from sklearn.metrics import accuracy_score
import sys
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
#import xgboost as xgb
import multiprocessing
import joblib
from sklearn.ensemble import AdaBoostClassifier #as adb
from imblearn.metrics import classification_report_imbalanced
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


#traing dataset
x_train = pd.read_csv('Iot_Train_x_std.csv',header=None)
y_train = pd.read_csv('Iot_Train_y.csv',header=None)

#split dataset for train and test
train_x, test_x, train_Y, test_Y = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

# # train_Y[0]=list(train_Y[0].map({'normal':0, 'dos':1, 'ddos':2, 'injection':3, 'xss':4, \
#                                 'password':5, 'scanning':6, 'mitm':7}))
# test_Y[0]=list(train_Y[0].map({'normal':0, 'dos':1, 'ddos':2, 'injection':3, 'xss':4, \
#                                 'password':5, 'scanning':6, 'mitm':7}))

#use few dataset to test 
print("train_x", train_x.head())
print("train_y", train_Y.head())
print("test_x", test_x.head())
print("test_y", test_Y.head())
print(train_x.info())
print(train_Y.info())
print(test_x.info())
print(test_Y.info())

train_Y = train_Y.values.ravel()
test_Y  = test_Y.values.ravel()    
    

# print('train dataset shape %s' % train_Y.value.count())
# print('test dataset shape %s' % test_Y.value.count())

print('train dataset shape %s' % Counter(train_Y))
print('test dataset shape %s' % Counter(test_Y))
# sys.exit(0)

###building adaboost model
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
target_names = ['normal 0', 'dos 1', 'ddos 2', 'injection 3', 'xss 4','password 5','scanning 6', 'mitm 7']

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

# ImbalancedLearning
Combine uncersampling and oversampling with ensemble learning

CIC-MalMem-2022 dataset (https://www.unb.ca/cic/datasets/malmem-2022.html)
NSL-KDD dataset  (https://www.unb.ca/cic/datasets/nsl.html )
TON_IoT dataset (https://research.unsw.edu.au/projects/toniot-datasets)


Directories：

TON_IoT is for the code of TON_IoT dataset.

MalMem is for the code of CIC-MalMem-2022 dataset

NSL-KDD is for the code of NSL-KDD dataset

Data has been preprossed and normalized.

The code name begins with evaluate.... is for the Precision-Recall Curve and ROC curve evaluation

The data has been preprocessed and it is divided into traing set (70%) and test set(20%).

For example, the files in the folder "MalMem":

RUS+ADASYN+Tomek Link run on balanced and imbalanced data:

     adasyn_tl_xgboost_balance.ipynb
     adasyn_tl_xgboost_imbalanced.ipynb

Majority voting classifier (ensemble of Logistic Regression, Decision Tree, SVM, K-Nearest Neighbor) run on balanced and imbalanced data:

     ensemble_major.ipynb   (balanced)
     ensemble_major_imb.ipynb   (imbalanced)
     
CNN run on balanced and imbalanced data:

     createCNN_Balanced.ipynb
     createCNN_Imbalanced.ipynb
     
Precision-Recall Curve evaluation on balanced data classified by XGBoost, majority voting classifier, CNN

     evaluate_PR_xgboost.ipynb      (balanced)
     evaluate_PR_xgboost_imb.ipynb   (imbalanced)
     ......
     Evaluate_PR_3Algorithms.ipynb   (three algorithms comparison)
     
Area Under ROC compared by three algorithms, ROC curve was generated in the code to process balanced data

     evaluate_auc_3Algorithm_micro.ipynb
     
Test AdaBoost and RUSBoost:

     test_AdaBoost_MalMem.py   (Test AdaBoost and RUSBoost for CIC-MalMem-2022)
     test_AdaBoost_NSL-KDD.py, test_RUSAdaboost.ipynb    (Test AdaBoost and RUSBoost for NSL-KDD)
    
     test_AdaBoost_IoT.py     (Test AdaBoost and RUSBoost for TON_IoT)

The files in folder "NSL-KDD" and "TON_IoT" have the similiar name as the files in folder "MalMem"
     

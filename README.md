# ImbalancedLearning
Combine uncersampling and oversampling with ensemble learning
Directories：

TON_IoT is for the code of TON_IoT dataset

MalMem is for the code of CIC-MalMem-2022 dataset

NSL-KDD is for the code of NSL-KDD dataset

The code name begins with evaluate.... is for the Precision-Recall Curve and ROC curve evaluation

The data has been preprocessed and it is divided into traing set (70%) and test set(20%).

RUS+ADASYN+Tomek Link run on balanced and imbalanced data:

     adasyn_tl_xgboost_balance.ipynb
     adasyn_tl_xgboost_imbalanced.ipynb

Majority voting classifier (ensemble of Logistic Regression, Decision Tree, SVM, K-Nearest Neighbor) run on balanced and imbalanced data:

     ensemble_major.ipynb
     ensemble_major_imb.ipynb
     
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
     
     

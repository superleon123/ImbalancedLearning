# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:06:14 2022

@author: Administrator
"""
"""
processing Ton_IoT train and test data, and create csv files.
"""

import os, sys
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.preprocessing import StandardScaler #RobustScaler,MinMaxScaler
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

dataset_root = 'Ton_IoT/'
#train_file = os.path.join(dataset_root, 'KDDTrain+.txt')
#train_file = 'KDDTrain+.txt'
#test_file = os.path.join(dataset_root, 'KDDTest+.txt')
#test_file = 'KDDTest+.txt'

orig_file = 'Train_Test_Windows_10.csv'


# Original KDD dataset feature names obtained from 
# http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
# http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
"""
header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'success_pred']
# Differentiating between nominal, binary, and numeric features

# root_shell is marked as a continuous feature in the kddcup.names 
# file, but it is supposed to be a binary feature according to the 
# dataset documentation

col_names = np.array(header_names)

nominal_idx = [1, 2, 3]   # name column
binary_idx = [6, 11, 13, 14, 20, 21]    # 二进制column
numeric_idx = list(set(range(41)).difference(nominal_idx).difference(binary_idx))   #数值型column数据

nominal_cols = col_names[nominal_idx].tolist()
binary_cols = col_names[binary_idx].tolist()
numeric_cols = col_names[numeric_idx].tolist()

# training_attack_types.txt maps each of the 22 different attacks to 1 of 4 categories
# file obtained from http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types
"""
# create mapping for attack type number
category = defaultdict(list)
#category['benign'].append('normal')
category['0'].append('normal')

print("category:", category)

with open('train_attack_type_num.txt', 'r') as f:
    for line in f.readlines():
         # print(line)
         attack, cat = line.strip().split(',')
         print(attack, cat)
         category[cat].append(attack)
         
# print(category)
attack_mapping = dict((v,k) for k in category for v in category[k])
print ("attack_mapping",attack_mapping)
#print(train_file)
#print(test_file)

#read file
# train_df = pd.read_csv(train_file, names=header_names)
#train_df['attack_category'] = train_df['attack_type'] \
#                                .map(lambda x: attack_mapping[x])

orig_df = pd.read_csv(orig_file)
# print(orig_df.count())
#print(orig_df)


orig_df['attack_num'] = orig_df['type'].map(lambda x: attack_mapping[x])
print(orig_df.head())

#print('dataset shape %s' % Counter(orig_df))

#print(orig_df.describe())
#print(orig_df.groupby(['attack_num']).size())
print(orig_df.groupby(['type']).size())

# train_Y = orig_df['type']   #english name of attack type
train_Y = orig_df['attack_num']  #convert into number

# eliminate some columns with 0 value
orig_df.drop(['label', 'type','attack_num'], axis=1, inplace=True)
orig_df.drop(['Processor_pct_ C3_Time','Processor_pct_ C2_Time','Processor_C2_ransitions_sec', \
'Processor_C3_ransitions_sec','Process_Elapsed_Time', \
'Process_Creating Process ID','Process_ID Process', 'Process_Priority Base', \
'Network_I(Intel R _82574L_GNC) Packets Received Unknown', \
'Network_I(Intel R _82574L_GNC) Packets Outbound Errors', \
'Network_I(Intel R _82574L_GNC) Packets Received Discarded', \
'Network_I(Intel R _82574L_GNC) Packets Outbound Discarded', \
'Network_I(Intel R _82574L_GNC) TCP RSC Exceptions sec', \
'Network_I(Intel R _82574L_GNC) Output Queue Length', \
'Network_I(Intel R _82574L_GNC) Packets Sent Non-Unicast sec', \
'Network_I(Intel R _82574L_GNC) Packets Received Non-Unicast sec', \
'Network_I(Intel R _82574L_GNC) TCP RSC Coalesced Packets sec', \
'Network_I(Intel R _82574L_GNC) Offloaded Connections', \
'Network_I(Intel R _82574L_GNC) Packets Received Errors', \
'Memory System Code Resident Bytes', \
'LogicalDisk(_Total) Current Disk Queue Length', 'Network_I(Intel R _82574L_GNC) Current Bandwidth', \
'Process_Page Faults_sec',   #has null value
], axis=1, inplace=True)


# orig_df.isnull()
# orig_df.dropna(how='any', inplace=True)  #删除有缺失值的行

# sys.exit(0)

#print("sum_null", orig_df.isnull().sum())


print("orig_df",orig_df.info())
# sys.exit(0)
print("orig_df",orig_df.dtypes)
# sys.exit(0)

# orig_df_conv=np.array(orig_df,dtype=np.float)
# orig_df_conv=pd.DataFrame(orig_df_conv)


# print("orig_df_conv",orig_df_conv.info())
# print("orig_df_conv",orig_df_conv.dtypes)
# sys.exit(0)


# # train_x_imp = orig_df

# # from sklearn.impute import SimpleImputer
# # imputer = SimpleImputer(strategy="median")
# # imputer.fit(train_x_imp)
# # print(imputer.statistics_)
# # train_x_2 = imputer.transform(train_x_imp)
# # train_x = pd.DataFrame(train_x_2, colums=train_x_imp.columns)

# train_x = orig_df['Process_IO Read_Operations_sec'].fillna(0)

# train_x = orig_df['Process_IO Read_Operations_sec']

#train_x = orig_df
# print("train_x", train_x.head())
#print("train_x", train_x.describe().transpose())
#print(train_df[binary_cols].describe().transpose())
# sys.exit(0)

# data clearning 
# eliminate space in string, and assign NaN value for object type columns
# def data_cleaning2(data,*cols):
def data_cleaning_del_space(data):
    cols = data.columns.tolist()   #leon add
    for c in cols:
        if data[c].dtype == 'object':  #leon add
            data[c] = data[c].str.replace(' ','')
    return data

# train_x = data_cleaning2(orig_df,'Process_IO Read_Operations_sec')
train_x = data_cleaning_del_space(orig_df)


def data_cleaning_give_Nan(data):
    cols = data.columns.tolist()
    for col in cols:
        if data[col].dtype == 'object':
            data[col].fillna('NaN',inplace=True ) # inplace=True,表在原数据上进行更改
        else:
            data[col].fillna(0,inplace=True)
    return data

train_x = data_cleaning_give_Nan(train_x)
    
print("train_x info", train_x.info())
"""
test_df = pd.read_csv(test_file, names=header_names)
test_df['attack_category'] = test_df['attack_type'].map(lambda x: attack_mapping[x])
test_df.drop(['success_pred'], axis=1, inplace=True)

train_attack_types = train_df['attack_type'].value_counts()
train_attack_cats = train_df['attack_category'].value_counts()

test_attack_types = test_df['attack_type'].value_counts()
test_attack_cats = test_df['attack_category'].value_counts()
"""



"""
train_attack_types.plot(kind='barh', figsize=(20,10), fontsize=20)
train_attack_cats.plot(kind='barh', figsize=(20,10), fontsize=30)

test_attack_types.plot(kind='barh', figsize=(20,10), fontsize=15)
test_attack_cats.plot(kind='barh', figsize=(20,10), fontsize=30)
"""
"""
print(train_df[binary_cols].describe().transpose())
print(train_df.groupby(['su_attempted']).size())
# Let's fix this discrepancy and assume that su_attempted=2 -> su_attempted=0
train_df['su_attempted'].replace(2, 0, inplace=True)
test_df['su_attempted'].replace(2, 0, inplace=True)
print(train_df.groupby(['su_attempted']).size())

# Next, we notice that the num_outbound_cmds column only takes on one value!

print(train_df.groupby(['num_outbound_cmds']).size())
# Now, that's not a very useful feature - let's drop it from the dataset
train_df.drop('num_outbound_cmds', axis = 1, inplace=True)
test_df.drop('num_outbound_cmds', axis = 1, inplace=True)
numeric_cols.remove('num_outbound_cmds')

#train_df.count()
#test_df.count()

print(train_df.groupby(['attack_category']).size())
print(train_df.count())
print(test_df.groupby(['attack_category']).size())
print(test_df.count())
sys.exit(0)

# Data preparation
train_Y = train_df['attack_category']
train_x_raw = train_df.drop(['attack_category','attack_type'], axis=1)
test_Y = test_df['attack_category']
test_x_raw = test_df.drop(['attack_category','attack_type'], axis=1)

combined_df_raw = pd.concat([train_x_raw, test_x_raw])
combined_df = pd.get_dummies(combined_df_raw, columns=nominal_cols, drop_first=True)

print("len(train_x_raw)", len(train_x_raw))
train_x = combined_df[:len(train_x_raw)]   #select train dataset 
test_x = combined_df[len(train_x_raw):]    #select test dataset

# Store dummy variable feature names
dummy_variables = list(set(train_x)-set(combined_df_raw))
print(train_x.describe())
#train_x['duration'].describe()
"""




#output train and test dataset to csv and read train set for standardization and ouput
#to csv again
def output_pr_data():  
    train_x.to_csv('Iot_Train_x.csv',header=None,index=None)
    train_Y.to_csv('Iot_Train_y.csv',header=None,index=None)
    # test_x.to_csv('KDDTest_x.csv',index=None,header=None)  
    # test_Y.to_csv('KDDTest_y.csv',index=None,header=None) 
    
    #train_x[numeric_cols].to_csv('Test_x.csv',index=None,header=None)
    # test_x[numeric_cols].to_csv('KDDTest_x_num.csv',index=None,header=None)  #numeric cols
    # test_x[numeric_cols].to_csv('KDDTest_x_num2.csv',index=None)  #numeric cols
    
    
    # train_x[numeric_cols].to_csv('KDDTrain_x_num.csv',header=None,index=None)
    # train_Y[numeric_cols].to_csv('KDDTrain_y_num.csv',header=None,index=None)
    
output_pr_data()



#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)



# Let's proceed with StandardScaler- Apply to train_x
# fill NaN value with mean value for all columns
# train_x_std = pd.read_csv('Iot_Train_x.csv',nrows=5,header=None)
train_x_std = pd.read_csv('Iot_Train_x.csv',header=None)
print("train_x_std---", train_x_std.info())
train_x_std.fillna(train_x_std.mean(), inplace=True)


# print("before standardization\n", train_x_std.describe())
# print("before standardization\n", train_x_std.head())
# train_x_std = data_cleaning2(train_x_std)


# print("train_x_std", train_x_std.info())
# print("train_x_std",train_x_std.dtypes)
# sys.exit(0)

# train_x_std2 = np.array(train_x_std,dtype=np.float)
# train_x_std2 = train_x_std.to_numpy(dtype=float, copy=False)
#train_x_std2 = pd.to_numeric(train_x_std)

# train_x_std2 = train_x_std.convert_dtypes()

# train_x_std2 = train_x_std.copy()
# train_x_std2 = train_x_std2.astype(float)

# train_x_std2['1'] = train_x_std2['1'].astype(float) 
# train_x_std2 = train_x_std2.astype({'1': float, '2': float})

# train_x_std2 = train_x_std2.astype(float)

# print("train_x_std2", train_x_std2.info())
# print("train_x_std2",train_x_std2.dtypes)
#train_x_std3 = pd.DataFrame(train_x_std2)


# print("orig_df_conv",orig_df_conv.info())
# print("orig_df_conv",orig_df_conv.dtypes)
# sys.exit(0)


# # train_x_imp = orig_df

# # from sklearn.impute import SimpleImputer
# # imputer = SimpleImputer(strategy="median")
# # imputer.fit(train_x_imp)
# # print(imputer.statistics_)
# # train_x_2 = imputer.transform(train_x_imp)
# # train_x = pd.DataFrame(train_x_2, colums=train_x_imp.columns)

# sys.exit(0)
#standization
standard_scaler = StandardScaler().fit(train_x_std)
# standard_scaler = MinMaxScaler().fit(train_x_std)
train_x_std_after = standard_scaler.transform(train_x_std)

train_x_std_after = pd.DataFrame(train_x_std_after)

print("train_x_std_after:",train_x_std_after.info())
#print(train_x_std3.dtypes)
#save to a file for ML
train_x_std_after.to_csv('Iot_Train_x_std.csv',header=None,index=None)



"""
# 5-class classification version, Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, zero_one_loss, \
    accuracy_score, classification_report

classifier = DecisionTreeClassifier(random_state=17)
classifier.fit(train_x, train_Y)

pred_y = classifier.predict(test_x)

results = confusion_matrix(test_Y, pred_y, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=results, display_labels=classifier.classes_)
error = zero_one_loss(test_Y, pred_y)
print(results)
print("zero_one_loss:", error)

disp.plot()
plt.show()
print("accuracy_score", accuracy_score(test_Y, pred_y))
print(classification_report(test_Y, pred_y, target_names=classifier.classes_))

"""
"""
#KNearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, zero_one_loss, \
    accuracy_score, classification_report
    
classifier = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
classifier.fit(train_x, train_Y)

pred_y = classifier.predict(test_x)

results = confusion_matrix(test_Y, pred_y)
error = zero_one_loss(test_Y, pred_y)

print(results)
print(error)
"""


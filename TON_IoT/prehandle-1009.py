# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 23:56:15 2021

@author: Administrator
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder  
from pandas.core.frame import DataFrame
import numpy as np
from sklearn import preprocessing

def get_total_data():  #定义数值替换文本的函数
    data = pd.read_csv('10_percent_corrected.csv',header=None)   #header=None
    print("csv data \n", data.head())
    #将源文件中3种协议类型转换成数字标识
    data[2]=list(data[2].map({'tcp':0, 'udp':1, 'icmp':2}))   #leon origin: data[1]
    #print("test2:", data[2])
    
    #将源文件中70种网络服务类型转换成数字标识  origin: data[2]
    data[3]=list(data[3].map({'aol':0, 'auth':1, 'bgp':2, 'courier':3, 'csnet_ns':4,\
                              'ctf':5, 'daytime':6, 'discard':7, 'domain':8, 'domain_u':9,'echo':10, 'eco_i':11, 'ecr_i':12, 'efs':13, 'exec':14,'finger':15, 'ftp':16, 'ftp_data':17, 'gopher':18, 'harvest':19,'hostnames':20, 'http':21, 'http_2784':22, 'http_443':23, 'http_8001':24,'imap4':25, 'IRC':26, 'iso_tsap':27, 'klogin':28, 'kshell':29,'ldap':30, 'link':31, 'login':32, 'mtp':33, 'name':34,'netbios_dgm':35, 'netbios_ns':36, 'netbios_ssn':37, 'netstat':38, 'nnsp':39,'nntp':40, 'ntp_u':41, 'other':42, 'pm_dump':43, 'pop_2':44,'pop_3':45, 'printer':46, 'private':47, 'red_i':48, 'remote_job':49,'rje':50, 'shell':51, 'smtp':52, 'sql_net':53, 'ssh':54,'sunrpc':55, 'supdup':56, 'systat':57, 'telnet':58, 'tftp_u':59,'tim_i':60, 'time':61, 'urh_i':62, 'urp_i':63, 'uucp':64,'uucp_path':65, 'vmnet':66, 'whois':67, 'X11':68, 'Z39_50':69}) )
    #print("test3:", data[3])
  
    #将源文件中11种网络连接状态转换成数字标识 origin: data[3]
    data[4]=list(data[4].map({'OTH':0, 'REJ':0, 'RSTO':0,'RSTOS0':0, 'RSTR':0, 'S0':0,'S1':0, 'S2':0, 'S3':0,'SF':1, 'SH':0}))
    #将源文件中攻击类型转换成数字标识(训练集中共出现了22个攻击类型，而剩下的17种只在测试集中出现)
    #print("test4:", data[4])
    
    #data[5] = pd.Series(data[5]).astype(float)
    
    #convert columns to int
    #data[5] = data[5].astype('float')
    data[5] = data[5].apply(to_int) 
    data[6] = data[6].apply(to_int) 
    #data[5] = data[5].apply(convert_data) 
    data[23] = data[23].apply(to_int)
    data[24] = data[24].apply(to_int)
    data[32] = data[32].apply(to_int)
    data[33] = data[33].apply(to_int)
    
    ################
    
    #origin: 41
    data[42]=list(data[42].map({'normal.':0, 'ipsweep.':1, 'mscan.':2, 'nmap.':3, 'portsweep.':4, 'saint.':5, 'satan.':6, 'apache2.':7,'back.':8, 'land.':9, 'mailbomb.':10, 'neptune.':11, 'pod.':12,'processtable.':13, 'smurf.':14, 'teardrop.':15, 'udpstorm.':16, 'buffer_overflow.':17, 'httptunnel.':18, 'loadmodule.':19, 'perl.':20, 'ps.':21,'rootkit.':22, 'sqlattack.':23, 'xterm.':24, 'ftp_write.':25,'guess_passwd.':26, 'imap.':27, 'multihop.':28, 'named.':29, 'phf.':30,'sendmail.':31, 'snmpgetattack.':32, 'snmpguess.':33, 'spy.':34, 'warezclient.':35,'warezmaster.':36, 'worm.':37, 'xlock.':38, 'xsnoop.':39}))
    #print("test42:", data[42])
    
    #数值归一化:最值归一化           
    #data[3] = (data[3]-data[3].min())/(data[3].max() - data[3].min())
    #print("***test:", data[4])
    #data[5] = int(data[5])
    
    #print("test5:", data[5])
    
    #data[5] = (data[5]-data[5].min())/(data[5].max() - data[5].min())
   # print("data type", data[5].dtype())
    
    #data[5] = data[5].astype('int')
    #data[5] = list(data[5])
    
    #print(data.info())
    #print("after convert:", data[5])
    #aa= type(data[5])
    #aa= data[5].max()
    #aaa=data[5].min()
    #print("aa&aaa:", aaa,aa)
   # data[5] = (data[5]-data[5].min())   #/(data[5].max() - data[5].min())
    
    
    #data[6] = (data[6]-data[6].min())/(data[6].max() - data[6].min())
    #data[23] = (data[23]-data[23].min())/(data[23].max() - data[23].min())
    #data[24] = (data[24]-data[24].min())/(data[24].max() - data[24].min())
    #data[32] = (data[32]-data[32].min())/(data[32].max() - data[32].min())
    #data[33] = (data[33]-data[33].min())/(data[33].max() - data[33].min())
    
    #np.isnan(data).any()  #data里是否存在nan, will raise error, due to string and numeric types both exist
    
    #np.isnan(data.astype(np.float).any()
        
    
    data.isnull()
    data.dropna(how='any', inplace=True)  #删除有缺失值的行
    #data.dropna(inplace=True)  #删除有缺失值的行
    
    return data
    
    
def get_target_data(): #定义标签独热编码的函数
    data = get_total_data()
    # enc = OneHotEncoder(sparse = False) #独热编码  sparse=False 直接生成array对象
    # enc.fit([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39]]) #有40种元素，说明用40个状态位来表示
    # result = enc.transform(data[[42]]) #就是将data[41]这个特征转换成one-hot编码 origin:data[41]
    # return DataFrame(result)
    data = data.iloc[:, 42]
    data = data.astype(int)
    print("target data \n", data.head())
    return data  

def get_input_data(): #返回x   train set
    data = get_total_data()
   # del data[42]  #origin: data[42]
     
    data = data.iloc[:, 1:42]
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled_data = minmax_scale.fit_transform(data)
    
#    print("train data \n", scaled_data.head())
    return scaled_data


def convert_data(value):
    """
    转换字符串数字为float类型
     - 移除 ￥ ,
     - 转化为float类型
    """
    #new_value = value.replace(',', '').replace('￥', '')
    new_value = int(value)
    return new_value


def to_int(str):
    try:
        int(str)
        return int(str)
    except ValueError: #报类型错误，说明不是整型的
        try:
            float(str) #用这个来验证，是不是浮点字符串
            return int(float(str))
        except ValueError:  #如果报错，说明即不是浮点，也不是int字符串。   是一个真正的字符串
            return False


if __name__ == '__main__':
    data_input = get_input_data()    #获取x
    data_input2 = pd.DataFrame(data_input)
    data_input2.to_csv('train_2.csv',header=None,index=None)
    data_target = get_target_data()  #获取标签
    data_target.to_csv('test_2.csv',index=None,header=None)

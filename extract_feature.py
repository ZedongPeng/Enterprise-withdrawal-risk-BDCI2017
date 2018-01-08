
import pandas as pd
import numpy as np
import datetime  
import numpy as np  
from collections import Counter
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
import re
from sklearn.cluster import KMeans
np.set_printoptions(threshold=np.inf) 

# data preprocess
############################################  load data  #####################################################
entbase = pd.read_csv('data/1entbase.csv')#(255131,10)
entbase['EID'] = entbase['EID'].astype('str')
entbase['RGYEAR'] = entbase['RGYEAR'].astype('str')
entbase['HY'] = entbase['HY'].astype('str')
entbase['ZCZB'] = entbase['ZCZB'].astype('float')
entbase['ETYPE'] = entbase['ETYPE'].astype('str')
entbase['MPNUM'] = entbase['MPNUM'].astype('float')
entbase['INUM'] = entbase['INUM'].astype('float')
entbase['ENUM'] = entbase['ENUM'].astype('float')
entbase['EID'] = entbase['EID'].astype('str')
entbase['FINZB'] = entbase['FINZB'].astype('float')
entbase['FSTINUM'] = entbase['FSTINUM'].astype('float')
entbase['TZINUM'] = entbase['TZINUM'].astype('float')
entbase = entbase.drop_duplicates(['EID'])

alter = pd.read_csv('data/2alter.csv')#(302105, 5)
alter['EID'] = alter['EID'].astype('str')
alter = alter.drop_duplicates(['EID', 'ALTERNO', 'ALTDATE'])


branch = pd.read_csv('data/3branch.csv')#(107844, 5)
branch.columns = ['EID','TYPECODE','IFHOME','B_REYEAR','B_ENDYEAR']
branch['EID'] = branch['EID'].astype('str')
branch = branch.drop_duplicates(['EID', 'TYPECODE'])

invest = pd.read_csv('data/4invest.csv')#(55526, 6)
invest.columns = ['EID','BTEID','IFHOME','BTBL','BTYEAR','BTENDYEAR']
invest['EID'] = invest['EID'].astype('str')
invest = invest.drop_duplicates(['EID', 'BTEID'])

right = pd.read_csv('data/5right.csv')#(1118502, 5)
right.columns = ['EID','RIGHTTYPE','TYPECODE','ASKDATE','FBDATE']
right['EID'] = right['EID'].astype('str')
right['RIGHTTYPE'] = right['RIGHTTYPE'].astype('str')
#right = right.drop_duplicates(['EID', 'RIGHTTYPE', 'TYPECODE'])

project = pd.read_csv('data/6project.csv')#(32827, 4)
project.columns = ['EID','TYPECODE','DJDATE','IFHOME']
project['EID'] = project['EID'].astype('str')
project = project.drop_duplicates(['EID', 'TYPECODE'])

lawsuit = pd.read_csv('data/7lawsuit.csv')#(25544, 4)
lawsuit.columns = ['EID','TYPECODE','LAWDATE','LAWAMOUNT']
lawsuit['EID'] = lawsuit['EID'].astype('str')
lawsuit = lawsuit.drop_duplicates(['EID', 'TYPECODE'])

breakfaith = pd.read_csv('data/8breakfaith.csv')#(3658, 4)
breakfaith.columns = ['EID','TYPECODE','FBDATE','SXENDDATE']
breakfaith['EID'] = breakfaith['EID'].astype('str')
breakfaith = breakfaith.drop_duplicates(['EID', 'TYPECODE'])

recruit = pd.read_csv('data/9recruit.csv')#(31498, 4)
recruit.columns = ['EID','WZCODE','POSCODE','RECDATE','PNUM']
recruit['EID'] = recruit['EID'].astype('str')
recruit = recruit.drop_duplicates(['EID','WZCODE','POSCODE'])

qualification = pd.read_csv('data/10qualification.csv', encoding='latin-1')
qualification.columns = ['EID','ADDTYPE','BEGINDATE','EXPIRYDATE']
qualification['EID'] = qualification['EID'].astype('str')
qualification = qualification.drop_duplicates(['EID','ADDTYPE'])


evaluation_public = pd.read_csv('data/evaluation_public.csv',header=None)#(102125, 1)
evaluation_public.columns = ['EID']
evaluation_public = evaluation_public.drop(0).reset_index(drop = True)
evaluation_public['EID'] = evaluation_public['EID'].astype('str')

train = pd.read_csv('data/train.csv',header=None)
train.columns = ['EID', 'TARGET','ENDDATE']
train = train.drop(0).reset_index(drop = True)
train['EID'] = train['EID'].astype('str')
train['ENDDATE'] = train['ENDDATE'].fillna(0)
train['ENDDATE'] = train['ENDDATE'].astype('int')
train['TARGET'] = train['TARGET'].astype('int')
train_and_evaluation = pd.merge(train,evaluation_public, on = 'EID', how = 'outer')
train_and_evaluation = train_and_evaluation.drop('TARGET', axis = 1)



entbase_train_and_evaluation = pd.merge(entbase,train_and_evaluation,on='EID',how='inner')
alter_train_and_evaluation = pd.merge(alter,train_and_evaluation,on='EID',how='inner')
branch_train_and_evaluation = pd.merge(branch,train_and_evaluation,on='EID',how='inner')
invest_train_and_evaluation = pd.merge(invest,train_and_evaluation,on='EID',how='inner')
right_train_and_evaluation = pd.merge(right,train_and_evaluation,on='EID',how='inner')
project_train_and_evaluation = pd.merge(project,train_and_evaluation,on='EID',how='inner')
lawsuit_train_and_evaluation = pd.merge(lawsuit,train_and_evaluation,on='EID',how='inner')
breakfaith_train_and_evaluation = pd.merge(breakfaith,train_and_evaluation,on='EID',how='inner')
recruit_train_and_evaluation = pd.merge(recruit,train_and_evaluation,on='EID',how='inner')


###############################  extract data from traning set  ###################################
def peid2eid(x):
    if x[0] == 'p':
        return int(x[1:])
    else:
        return int(x[1:]) + 600000

def get_date(s):
	date = int(s[0:4])*12 + int(s[5:7])
	return date	


def get_month_gap_before(s):
	gap = 12*(2017-int(s[0:4])) + 8 - int(s[5:7])
	return gap

def get_year_gap_before(s):
	gap = 12*(2018-int(s))
	return gap

def delete_month(s):
	year,month = s.split('-')
	return year

def get_coefficient(X_parameters,Y_parameters):
	regr = linear_model.LinearRegression()
	regr.fit(X_parameters.reshape(len(X_parameters),1),Y_parameters.reshape(len(Y_parameters),1))
	return regr.coef_[0]

def get_Kmeans(s,t):
	cluster_kmeans = KMeans(n_clusters = t,random_state = 0).fit(s)
	return cluster_kmeans.labels_

def EID_Classify(s):
	if s[0] == 'p':
		result = 1
	elif s[0] =='s':
		result =0
	return result

#------------------------------feature from 1entbase------------------------------------------
#1: RGYEAR 成立年度GAP
#2: HY 行业大类 one-hot编码
#3: ETYPE 企业类型 one-hot编码
#4: 注册资本和各种身份指标 'ZCZB', 'MPNUM', 'INUM', 'ENUM','FINZB', 'FSTINUM', 'TZINUM'
#5: 注册资本onehot编码
#6: FINZB/ZCZB   FINZB+ZCZB
#7: ZCZB用KMeans聚类
#8: 省份类别
#9 PROV独热编码
#10 EID数值


#1: RGYEAR 成立年度GAP
t = entbase_train_and_evaluation[['EID','RGYEAR']]
t['RGYEAR'] = 2017 - t['RGYEAR'].astype('int')

#2: HY 行业大类 one-hot编码
t1 = entbase_train_and_evaluation[['EID','HY']]
HY_dummies = pd.get_dummies(t1.HY,prefix='HY')
t1['HY'] = t1['HY'].map(lambda x:"0" if x=="nan" else x )
t1['HY'] = t1['HY'].astype('float')
t1['HY'] = t1['HY'].astype('int64')
hy_lst = list(t1['HY'])
sorted_hy_lst = sorted(list(map(lambda x: (x[1], x[0]), list(Counter(hy_lst).items()))), reverse=True)
encode_lst = [30 for i in range(100)]
for i in range(30):
    encode_lst[sorted_hy_lst[i][1]] = i
encoded_hy = np.zeros(len(t1['EID']), dtype=np.int32)
for i in range(len(t1['EID'])):
    encoded_hy[i] = encode_lst[t1['HY'][i]]
t1['ENCODED_HY'] = pd.Series(encoded_hy, index=t1.index)
encode_hy =t1[['EID','ENCODED_HY']].copy()
t1 = pd.concat([t1,HY_dummies], axis = 1)


#3: ETYPE 企业类型 one-hot编码
t2 = entbase_train_and_evaluation[['EID','ETYPE']]
ETYPE_dummies = pd.get_dummies(t2.ETYPE,prefix='ETYPE')
t2 = pd.concat([t2.drop('ETYPE',axis = 1),ETYPE_dummies], axis = 1)

#4: 注册资本和各种身份指标 'ZCZB', 'MPNUM', 'INUM', 'ENUM','FINZB', 'FSTINUM', 'TZINUM'
t3 = entbase_train_and_evaluation[['EID', 'ZCZB', 'MPNUM', 'INUM', 'ENUM','FINZB', 'FSTINUM', 'TZINUM', 'PROV']]
TZINUM_dummies = pd.get_dummies(t3.TZINUM,prefix='TZINUM')
t3 = pd.concat([t3,TZINUM_dummies], axis = 1)
t3['INUM-ENUM'] = t3.INUM - t3.ENUM
t3['TZINUM'] = t3['TZINUM'].fillna(0)
t3['FSTINUM-TZINUM'] = t3['FSTINUM'] - t3['TZINUM']
t3['FSTINUM+TZINUM'] = t3['FSTINUM'] + t3['TZINUM']
t3['ENUM+TZINUM'] = t3['ENUM'] + t3['TZINUM']
t3['ENUM-TZINUM'] = t3['ENUM'] - t3['TZINUM']
t3['FSTINUM-MPNUM'] = t3['FSTINUM'] - t3['MPNUM']
t3['MPNUM+ENUM+FSTINUM+TZINUM'] = t3['MPNUM'] + t3['ENUM'] + t3['FSTINUM'] + t3['TZINUM']
t3['log_ZCZB'] = np.log1p(t3['ZCZB'])
t3['log_MPNUM'] = np.log1p(t3['MPNUM'])
t3['log_INUM'] = np.log1p(t3['INUM'])
#t3 = entbase_train_and_evaluation[['EID', 'MPNUM', 'INUM', 'FINZB', 'FSTINUM', 'TZINUM']]

#5: 注册资本onehot编码
t4 = entbase_train_and_evaluation[['EID','ZCZB']]
t4['ZCZB'] = t4['ZCZB'].fillna(0)
t4['ZCZB1'] = t4.ZCZB.apply(lambda x: 1 if x <= 1 else 0)
t4['ZCZB10'] = t4.ZCZB.apply(lambda x: 1 if 1 < x <= 10 else 0)
t4['ZCZB100'] = t4.ZCZB.apply(lambda x: 1 if 10 < x <= 100 else 0)
t4['ZCZB1000'] = t4.ZCZB.apply(lambda x: 1 if 100 < x <= 1000 else 0)
t4['ZCZB10000'] = t4.ZCZB.apply(lambda x: 1 if 1000 < x <= 10000 else 0)
t4['ZCZB100000'] = t4.ZCZB.apply(lambda x: 1 if 10000 < x <= 100000 else 0)
t4['ZCZB1000000'] = t4.ZCZB.apply(lambda x: 1 if 100000 < x <= 1000000 else 0)
t4['ZCZB10000000'] = t4.ZCZB.apply(lambda x: 1 if x > 1000000 else 0)
t4 = t4.drop('ZCZB',axis = 1)

#6: FINZB/ZCZB   FINZB+ZCZB
t5 = entbase_train_and_evaluation[['EID','ZCZB','FINZB']]
t5['ZCZB'] = t5['ZCZB'].fillna(1)
t5['FINZB'] = t5['FINZB'].fillna(0)
t5['ZCZB_FINZB_rate'] = entbase_train_and_evaluation.FINZB/entbase_train_and_evaluation.ZCZB
t5['ZCZB_FINZB_plus'] = entbase_train_and_evaluation.FINZB+entbase_train_and_evaluation.ZCZB
t5 = t5.drop(['ZCZB','FINZB'], axis = 1)

# t6 = entbase_train_and_evaluation[['EID']]
# t6['EID2'] = (t6['EID'].astype('int')/50000).astype('int')
# EID_dummies = pd.get_dummies(t6.EID2)
# t6 = pd.concat([t6.drop('EID2',axis = 1),ETYPE_dummies], axis = 1)

#7: ZCZB用KMeans聚类
t7 = entbase_train_and_evaluation[['EID','ZCZB']]
t7['ZCZB'] = t7['ZCZB'].fillna(0)
t7['ZCZB_KMeans'] = get_Kmeans(t7[['ZCZB']],8)
t7.to_csv('data/ZCZB_KMeans.csv',index = None)
t7 = t7.drop('ZCZB', axis = 1)

#8: 省份类别
t8 = entbase_train_and_evaluation[['EID']]
t8['EID_CLASS'] = t8.EID.apply(lambda x : EID_Classify(x))

#9 PROV独热编码
t9 = entbase_train_and_evaluation[['EID','PROV']]
PROV_dummies = pd.get_dummies(t9.PROV,prefix='PROV')
t9 = pd.concat([t9,PROV_dummies], axis = 1)

#10 EID数值
t10 = entbase_train_and_evaluation[['EID']]
t10['EID_NUMBER'] = t10.EID.apply(lambda x : x[1:])

feature_entbase = pd.merge(t,t1,on='EID')
feature_entbase = pd.merge(feature_entbase,t2,on=['EID'])
feature_entbase = pd.merge(feature_entbase,t3,on=['EID'])
#feature_entbase = pd.merge(feature_entbase,t4,on=['EID'])
feature_entbase = pd.merge(feature_entbase,t5,on=['EID'])
# feature_entbase = pd.merge(feature_entbase,t6,on=['EID'])
feature_entbase = pd.merge(feature_entbase,t7,on=['EID'])
feature_entbase = pd.merge(feature_entbase,t8,on=['EID'])
feature_entbase = pd.merge(feature_entbase,t9,on=['EID'])
feature_entbase = pd.merge(feature_entbase,t10,on=['EID'])
feature_entbase.to_csv('data/feature_entbase.csv',index = None)
print('entbase succeed')

#------------------------------feature from 2alter------------------------------------------
#['EID', 'ALTERNO', 'ALTDATE', 'ALTBE', 'ALTAF']

#1: 企业变更次数统计
#2: 变更事项代码one-hot编码
#3：2013-2015年变更次数的统计/2013-2015年变更次数趋势
#4：ALTAF/ALTBE比值
#5：美元、港币、人民币分类

#0: 企业变更次数统计
t = alter_train_and_evaluation[['EID']]
t['alter_count'] = 1
t = t.groupby('EID').agg('sum').reset_index()

#1: 变更事项代码one-hot编码
t1 = alter_train_and_evaluation[['EID','ALTERNO']]
ALTERNO_dummies = pd.get_dummies(t1.ALTERNO,prefix='ALTERNO')
t1 = pd.concat([t1,ALTERNO_dummies], axis = 1)
t1 = t1.groupby('EID').agg('sum').reset_index()

#2: 变更时间月度GAP
t2 = alter_train_and_evaluation[['EID','ALTDATE']]
t2.ALTDATE = t2.ALTDATE.apply(get_month_gap_before)
temp1 =t2.groupby('EID').agg('min').reset_index()
temp2 = t2.groupby('EID').agg('max').reset_index()
t2 = t2.groupby('EID').agg('mean').reset_index()
t2.rename(columns={'ALTDATE':'ALTDATE_GAP'},inplace=True)
t2['ALTDATE_GAP_MIN'] = pd.Series(temp1.iloc[:,1], index=t2.index)
t2['ALTDATE_GAP_Max'] = pd.Series(temp2.iloc[:,1], index=t2.index)


#3：2013-2015年变更次数的统计/2013-2015年变更次数趋势
t3 = alter_train_and_evaluation[['EID','ALTDATE']]
t3['ALTDATE_YEAR'] = t3['ALTDATE'].map(lambda x:x.split('-')[0]) 
alter_ALTDATE = pd.get_dummies(t3['ALTDATE_YEAR'])
X_parameters = np.array(alter_ALTDATE.axes[1]).astype('int')
alter_ALTERNO_merge = pd.concat([t3['EID'],alter_ALTDATE],axis=1)
t3 = alter_ALTERNO_merge.groupby(['EID']).sum().reset_index()
t3.rename(columns = {'2013':'ALTDATE_2013','2014':'ALTDATE_2014','2015':'ALTDATE_2015'},inplace=True)
t3['rate'] = t3['ALTDATE_2013'].astype('int') + t3['ALTDATE_2015'].astype('int') - 2*t3['ALTDATE_2014'].astype('int')
t3['rate2'] = t3[['ALTDATE_2013','ALTDATE_2014','ALTDATE_2015']].apply(lambda x: get_coefficient(X_parameters,x),axis = 1).iloc[:,0].round(2)

#4：ALTAF/ALTBE比值
t4 = alter_train_and_evaluation[['EID','ALTBE','ALTAF']]
t4['ALTBE'].fillna('0', inplace = True)
t4['ALTBE'] = t4['ALTBE'].apply(lambda x: re.findall(r'[-+]?\d*\.\d+|\d+', str(x))[0])
t4['ALTAF'].fillna(0, inplace = True)
t4['ALTAF'] = t4['ALTAF'].apply(lambda x: re.findall(r'[-+]?\d*\.\d+|\d+', str(x))[0])
t8 =t4.copy()
t4['ALTAF'] = t4['ALTAF'].fillna(0).astype('float')
t4['ALTBE'] = t4['ALTBE'].fillna(1).astype('float')
t4['ALTAF_ALTBE_RATE'] = (t4['ALTAF']/t4['ALTBE']).astype('float')
#t4['ALTBE'] = alter_train_and_evaluation[['ALTBE']].fillna(0).astype('float')
t4 = t4.groupby('EID').agg('max').reset_index()
t4.fillna(-1,inplace = True)

#5：美元、港币、人民币分类
alter1 = pd.read_csv('data/2alter_1.csv')#(302105, 5)
alter1 = alter1.drop_duplicates(['EID', 'ALTERNO', 'ALTDATE'])
t5 = alter1[['EID','ALTBE_TYPE_0','ALTBE_TYPE_1','ALTBE_TYPE_2','ALTAF_TYPE_0','ALTAF_TYPE_1','ALTAF_TYPE_2','ALTBE_1','ALTAF_1']]
t5 = t5.groupby('EID').agg('sum').reset_index()

#6
t6 = alter_train_and_evaluation[['EID','ALTDATE']]
t6.ALTDATE = t6.ALTDATE.apply(get_date)
temp1 = t6.groupby('EID').agg('min').reset_index()
temp2 = t6.groupby('EID').agg('max').reset_index()
t6 = t6.groupby('EID').agg('mean').reset_index()
t6['ALTER_date_min'] = pd.Series(temp1.iloc[:,1], index=t6.index)
t6['ALTER_date_max'] = pd.Series(temp2.iloc[:,1], index=t6.index)

#7 alter_year - rgyear
t7 = alter[['EID','ALTDATE']]
rgyear = entbase[['EID','RGYEAR']]
t7 = pd.merge(t7,rgyear,on='EID')
t7['alter_rgyear'] = t7['ALTDATE'].map(lambda x:x[:4]).astype(int) -t7['RGYEAR'].astype(int)
t7 = t7.drop(['ALTDATE','RGYEAR'],axis=1)
temp = t7.groupby('EID').agg('min').reset_index()
t7 = t7.groupby('EID').agg('max').reset_index()
t7.rename(columns={'alter_rgyear':'alter_rgyear_max'},inplace=True)
t7['alter_rgyear_min'] = pd.Series(temp.iloc[:,1], index=t7.index)

#8:  ALTAF-ALTBE
t8['ALTAF'] = t4['ALTAF'].fillna(0).astype('float')
t8['ALTBE'] = t4['ALTBE'].fillna(0).astype('float')
t8['ALTAF-ALTBE'] = t8['ALTAF'] - t8['ALTBE']
t8 = t8.drop(['ALTAF','ALTBE'],axis=1)
temp1 = t8.groupby('EID').agg('sum').reset_index()
temp2 = t8.groupby('EID').agg('min').reset_index()
temp3 = t8.groupby('EID').agg('max').reset_index()
t8 = t8.groupby('EID').agg('mean').reset_index()
t8['ALTAF-ALTBE_sum'] = pd.Series(temp1.iloc[:,1], index=t8.index)
t8['ALTAF-ALTBE_min'] = pd.Series(temp1.iloc[:,1], index=t8.index)
t8['ALTAF-ALTBE_max'] = pd.Series(temp1.iloc[:,1], index=t8.index)

feature_alter = pd.merge(t,t1,on='EID')
feature_alter = pd.merge(feature_alter,t2,on='EID')
feature_alter = pd.merge(feature_alter,t3,on='EID')
feature_alter = pd.merge(feature_alter,t4,on='EID')
feature_alter = pd.merge(feature_alter,t5,on='EID')
feature_alter = pd.merge(feature_alter,t6,on='EID')
feature_alter = pd.merge(feature_alter,t7,on='EID')
feature_alter.to_csv('data/feature_alter.csv',index = None)
print('alter succeed')

#------------------------------feature from 3branch------------------------------------------
#['EID', 'TYPECODE', 'IFHOME', 'B_REYEAR', 'B_ENDYEAR']
#1：分支企业数量统计
#2：分支机构在省内的数量和比例
#3：分支成立年度到2017的GAP
#4：分支成立年度到关停时的GAP
#5：分支survive的count和rate
#6：分支关停的count和rate
#7：分支成立趋势

#1：分支企业数量统计
t = branch_train_and_evaluation[['EID']]
t['this_ent_all_branches_count'] = 1
t = t.groupby('EID').agg('sum').reset_index()

#2：分支机构在省内的数量和比例
t1 = branch_train_and_evaluation[['EID', 'IFHOME']]
t1['IFHOME'] = t1['IFHOME'].fillna(0.5)
t1.IFHOME = t1.IFHOME.astype('int')
t1 = t1.groupby('EID').agg('sum').reset_index()
temp = branch_train_and_evaluation.groupby(by=['EID'])['IFHOME'].count().reshape(27717,1)
t1['this_ent_all_branches_in_home_prob'] = t1.IFHOME.reshape(27717,1)/temp
t1.columns = ['EID', 'this_ent_all_branches_in_home_count','this_ent_all_branches_in_home_prob']

#3：分支成立年度到2017的GAP
t2 = branch_train_and_evaluation[['EID','B_REYEAR']]
t2['B_REYEAR'] = t2['B_REYEAR'].fillna(-1)
t2.B_REYEAR = t2.B_REYEAR.apply(lambda x:None if x == -1 else get_year_gap_before(x))
temp1 = t2.groupby('EID').agg('min').reset_index()
temp2 = t2.groupby('EID').agg('max').reset_index()
t2 = t2.groupby('EID').agg('mean').reset_index()
t2.rename(columns={'B_REYEAR':'B_REYEAR_GAP'},inplace=True)
t2['B_REYEAR_GAP_MIN'] = pd.Series(temp1.iloc[:,1], index=t2.index)
t2['B_REYEAR_GAP_MAX'] = pd.Series(temp2.iloc[:,1], index=t2.index)

#4：分支成立年度到关停时的GAP
t6 = branch_train_and_evaluation[['EID','B_REYEAR','B_ENDYEAR']]
t6['B_REYEAR'] = t6['B_REYEAR'].fillna(2017)
t6['B_ENDYEAR'] = t6['B_ENDYEAR'].fillna(2017)
t6['REYEAR_ENDYEAR_GAP'] = t6['B_ENDYEAR'] - t6['B_REYEAR']
t6 = t6.drop(['B_REYEAR','B_ENDYEAR'], axis = 1 )
temp1 = t6.groupby('EID').agg('min').reset_index()
temp2 = t6.groupby('EID').agg('max').reset_index()
t6 = t6.groupby('EID').agg('mean').reset_index()
t6['branch_GAP_MIN'] = pd.Series(temp1.iloc[:,1], index=t6.index)
t6['branch_GAP_MAX'] = pd.Series(temp2.iloc[:,1], index=t6.index)

#5：分支survive的count和rate
t3 = branch_train_and_evaluation[['EID','B_ENDYEAR']]
t3['B_ENDYEAR'] = t3['B_ENDYEAR'].fillna(-1)
t3['B_ENDYEAR'] = t3['B_ENDYEAR'].apply( lambda x: 1 if x == -1 else 0)
temp_count = t3.groupby(by=['EID'])['B_ENDYEAR'].count().reshape(27717,1)
t3 = t3.groupby('EID').agg('sum').reset_index()
t3['B_ENDYEAR_on_count_prob'] = t3.B_ENDYEAR.reshape(27717,1)/temp_count
t3.columns = ['EID','B_ENDYEAR_on_count','B_ENDYEAR_on_count_prob']

#6：分支关停的count和rate
t4 = branch_train_and_evaluation[['EID','B_ENDYEAR']]
t4['B_ENDYEAR'] = t4['B_ENDYEAR'].fillna(-1)
t4['B_ENDYEAR'] = t4['B_ENDYEAR'].apply( lambda x: 0 if x == -1 else 1)
temp_count = t4.groupby(by=['EID'])['B_ENDYEAR'].count().reshape(27717,1)
t4 = t4.groupby('EID').agg('sum').reset_index()
t4['B_ENDYEAR_off_count_prob'] = t4.B_ENDYEAR.reshape(27717,1)/temp_count
t4.columns = ['EID','B_ENDYEAR_off_count','B_ENDYEAR_off_count_prob']

#7：分支成立趋势
t5 = branch_train_and_evaluation[['EID','B_REYEAR']]
branch_B_REYEAR = pd.get_dummies(t5['B_REYEAR'])
branch_B_REYEAR = branch_B_REYEAR.iloc[:,-10:]
X_parameters = np.array(branch_B_REYEAR.axes[1]).astype('int')
branch_B_REYEAR_merge = pd.concat([t5['EID'],branch_B_REYEAR],axis=1)
t5 = branch_B_REYEAR_merge.groupby(['EID']).sum().reset_index()
t5['slope'] = t5.drop('EID',axis = 1).apply(lambda x: get_coefficient(X_parameters,x),axis = 1).iloc[:,0].round(2)*10
t5 = t5[['EID','slope']]

#8:branch b_reyear-rgyear
t7 = branch[['EID','B_REYEAR']]
rgyear = entbase[['EID','RGYEAR']]
t7 = pd.merge(t7,rgyear,on='EID')
t7['branch_rgyear'] = t7['B_REYEAR'].astype(int) -t7['RGYEAR'].astype(int)
t7 = t7.drop(['B_REYEAR','RGYEAR'],axis=1)
temp = t7.groupby('EID').agg('min').reset_index()
t7 = t7.groupby('EID').agg('max').reset_index()
t7.rename(columns={'branch_rgyear':'branch_rgyear_max'},inplace=True)
t7['branch_rgyear_min'] = pd.Series(temp.iloc[:,1], index=t7.index)

#9:branch b_endyear-rgyear
t8 = branch[['EID','B_ENDYEAR']]
t8['B_ENDYEAR'] = t8['B_ENDYEAR'].fillna(0)
rgyear = entbase[['EID','RGYEAR']]
t8 = pd.merge(t8,rgyear,on='EID')
t8['branch_end_rgyear'] = t8['B_ENDYEAR'].astype(int) -t8['RGYEAR'].astype(int)
t8 = t8.drop(['B_ENDYEAR','RGYEAR'],axis=1)
temp = t8.groupby('EID').agg('min').reset_index()
t8 = t8.groupby('EID').agg('max').reset_index()
t8.rename(columns={'branch_end_rgyear':'branch_end_rgyear_max'},inplace=True)
t8['branch_end_rgyear_min'] = pd.Series(temp.iloc[:,1], index=t8.index)



feature_branch = pd.merge(t,t1,on='EID')
feature_branch = pd.merge(feature_branch,t2,on='EID')
feature_branch = pd.merge(feature_branch,t3,on='EID')
feature_branch = pd.merge(feature_branch,t4,on='EID')
feature_branch = pd.merge(feature_branch,t5,on='EID')
feature_branch = pd.merge(feature_branch,t6,on='EID')
feature_branch = pd.merge(feature_branch,t7,on='EID')
feature_branch = pd.merge(feature_branch,t8,on='EID')
feature_branch.to_csv('data/feature_branch.csv',index = None)
print('branch succeed')

#------------------------------feature from 4invest------------------------------------------
#1：投资企业数量统计
#2：投资企业在省内的数量统计(空值赋0.5),投资企业在省内的比例统计
#3：投资企业存活的个数/比例
#4：投资企业成立趋势
#5：持股总数
#6：平均持股比例
#7：被投资企业类型

#1：投资企业数量统计
t = invest_train_and_evaluation[['EID']]
t['this_ent_invest_ent_count'] = 1
t = t.groupby('EID').agg('sum').reset_index()

#2：投资企业在省内的数量统计(空值赋0.5),投资企业在省内的比例统计
t1 = invest_train_and_evaluation[['EID', 'IFHOME']]
t1['IFHOME'] = t1['IFHOME'].fillna(0.5)
t1.IFHOME = t1.IFHOME.astype('int')
t1 = t1.groupby('EID').agg('sum').reset_index()
temp_count = t1.groupby(by=['EID'])['IFHOME'].count().reshape(t1.groupby(by=['EID'])['IFHOME'].count().shape[0],1)
t1['this_ent_all_invest_in_home_prob'] = t1.IFHOME.reshape(t1.IFHOME.shape[0],1)/temp_count
t1.columns = ['EID', 'this_ent_all_invest_in_home_count','this_ent_all_invest_in_home_prob']

#3：投资企业存活的个数/比例
t2 = invest_train_and_evaluation[['EID','BTENDYEAR']]
t2['BTENDYEAR'] = t2['BTENDYEAR'].fillna(-1)
t2['BTENDYEAR'] = t2['BTENDYEAR'].apply( lambda x: 1 if x == -1 else 0)
t2 = t2.groupby('EID').agg('sum').reset_index()
t2['this_ent_all_invest_survive_prob'] = t2.BTENDYEAR/invest_train_and_evaluation.groupby(by=['EID'])['BTENDYEAR'].count()
t2.columns = ['EID', 'this_ent_all_invest_survive_count','this_ent_all_invest_survive_prob']

#4：投资企业成立趋势
t3 = invest_train_and_evaluation[['EID','BTYEAR']]
invest_BTYEAR = pd.get_dummies(t3['BTYEAR'])
invest_BTYEAR = invest_BTYEAR.iloc[:,-10:]
X_parameters = np.array(invest_BTYEAR.axes[1]).astype('int')
invest_BTYEAR_merge = pd.concat([t3['EID'],invest_BTYEAR],axis=1)
t3 = invest_BTYEAR_merge.groupby(['EID']).sum().reset_index()
t3['slope'] = t3.drop('EID',axis = 1).apply(lambda x: get_coefficient(X_parameters,x),axis = 1).iloc[:,0].round(2)*10
t3 = t3[['EID','slope']]

#5：持股总数
t4 = invest_train_and_evaluation[['EID','BTBL']]
t4.BTBL = t4.BTBL.astype('float')
t4 = t4.groupby('EID').sum().reset_index()

#6：平均持股比例
t5 = invest_train_and_evaluation[['EID','BTBL']]
t5.BTBL = t5.BTBL.astype('float')
t5 = t5.groupby('EID').agg('mean').reset_index()

#7：被投资企业类型
t6 = invest_train_and_evaluation[['EID','BTEID']]
t6['BTEID_S'] = t6.BTEID.apply(lambda x: 1 if x[0]=='s' else 0)
t6['BTEID_P'] = t6.BTEID.apply(lambda x: 1 if x[0]=='p' else 0)
t6['BTEID_W'] = t6.BTEID.apply(lambda x: 1 if x[0]=='w' else 0)
t6 = t6.drop(['BTEID'],axis = 1)
t6 = t6.groupby('EID').agg('sum').reset_index()

#8:invest BTYEAR-rgyear
t7 = invest[['EID','BTYEAR']]
rgyear = entbase[['EID','RGYEAR']]
t7 = pd.merge(t7,rgyear,on='EID')
t7['invest_rgyear'] = t7['BTYEAR'].astype(int) -t7['RGYEAR'].astype(int)
t7 = t7.drop(['BTYEAR','RGYEAR'],axis=1)
temp = t7.groupby('EID').agg('min').reset_index()
t7 = t7.groupby('EID').agg('max').reset_index()
t7.rename(columns={'invest_rgyear':'invest_rgyear_max'},inplace=True)
t7['invest_rgyear_min'] = pd.Series(temp.iloc[:,1], index=t7.index)

feature_invest = pd.merge(t,t1,on='EID')
feature_invest = pd.merge(feature_invest,t2,on = 'EID')
feature_invest = pd.merge(feature_invest,t3,on = 'EID')
feature_invest = pd.merge(feature_invest,t4,on = 'EID')
feature_invest = pd.merge(feature_invest,t5,on = 'EID')
feature_invest = pd.merge(feature_invest,t6,on = 'EID')
feature_invest = pd.merge(feature_invest,t,on = 'EID')
feature_invest.to_csv('data/feature_invest.csv',index = None)
print('invest succeed')

#------------------------------feature from 5right------------------------------------------
#1：企业专利count
#2：权利类型ONE-HOT编码
#3：申请日期GAP
#4：权利申请到赋予的时间GAP
#5：right_typecode数字


#1：企业专利count
t = right[['EID']]
t['right_count'] = 1
t = t.groupby('EID').agg('sum').reset_index()

#2：权利类型ONE-HOT编码
t1 = right[['EID','RIGHTTYPE']]
RIGHTTYPE_dummies = pd.get_dummies(t1.RIGHTTYPE,prefix='RIGHTTYPE')
t1 = pd.concat([t1,RIGHTTYPE_dummies], axis = 1)
t1 = t1.groupby('EID').agg('sum').reset_index()

#3：申请日期GAP
t2 = right[['EID','ASKDATE']]
t2.ASKDATE = t2.ASKDATE.apply(get_month_gap_before)
temp1 = t2.groupby('EID').agg('min').reset_index()
temp2 = t2.groupby('EID').agg('max').reset_index()
t2 = t2.groupby('EID').agg('mean').reset_index()
t2.rename(columns={'ASKDATE':'ASKDATE_GAP'},inplace=True)
t2['ASKDATE_GAP_MIN'] = pd.Series(temp1.iloc[:,1], index=t2.index)
t2['ASKDATE_GAP_MAX'] = pd.Series(temp2.iloc[:,1], index=t2.index)

#4：权利申请到赋予的时间GAP
t3 = right[['EID','ASKDATE','FBDATE']].copy()
t3['ASKDATE'] = pd.to_datetime(t3['ASKDATE'])
t3['FBDATE'] = t3['FBDATE'].fillna('20170801')
t3['FBDATE'] = pd.to_datetime(t3['FBDATE'])
cc = []
for m in range(len(t3['ASKDATE'])):
	bb=t3['FBDATE'] [m] - t3['ASKDATE'] [m]
	cc.append(int(str(bb.days)))
t3['right_month_GAP'] = cc       
t3['right_month_GAP'] = (t3['right_month_GAP']/30).astype(int)                                        
t3 = t3.drop(['ASKDATE','FBDATE'],axis = 1)
#t3 = t3.sort_values('right_month_GAP',ascending=True).drop_duplicates('EID')
temp1 = t3.groupby('EID').agg('min').reset_index()
temp2 = t3.groupby('EID').agg('max').reset_index()
t3 = t3.groupby('EID').agg('mean').reset_index()
t3['right_month_GAP_MIN'] = pd.Series(temp1.iloc[:,1], index=t3.index)
t3['right_month_GAP_MAX'] = pd.Series(temp2.iloc[:,1], index=t3.index)

#5：right_typecode数字
t4 = right[['EID','TYPECODE']]
t4['TYPECODE'] = t4['TYPECODE'].apply(lambda x:x.lstrip('pno'))#去掉头部所有pno
t4['TYPECODE'] = t4['TYPECODE'].apply(lambda x:x.lstrip('mno'))
t4['TYPECODE'] = t4['TYPECODE'].apply(lambda x:x.lstrip('cno'))
t4['TYPECODE'] = t4['TYPECODE'].apply(lambda x:x.lstrip('GXB'))
t4.drop_duplicates('EID')
t4.rename(columns={'TYPECODE':'right_typecode'},inplace=True)
t4.right_typecode = t4.right_typecode.astype('int')

#6：right_typecode不同类型权利个数
t5 = right[['EID','TYPECODE']]
t5['right_pno'] = t5.TYPECODE.apply(lambda x: 1 if x.startswith('pno') else 0)
t5['right_mno'] = t5.TYPECODE.apply(lambda x: 1 if x.startswith('mno') else 0)
t5['right_cno'] = t5.TYPECODE.apply(lambda x: 1 if x.startswith('cno') else 0)
t5['right_GXB'] = t5.TYPECODE.apply(lambda x: 1 if x.startswith('GXB') else 0)
t5['right_number'] = t5.TYPECODE.apply(lambda x: 1 if x[:2].isdigit() else 0)
t5 = t5.drop(['TYPECODE'],axis = 1)
t5 = t5.groupby('EID').agg('sum').reset_index()

#7 right_date
t6 = right[['EID','ASKDATE']]
t6.ASKDATE = t6.ASKDATE.apply(get_date)
temp1 = t6.groupby('EID').agg('min').reset_index()
temp2 = t6.groupby('EID').agg('max').reset_index()
t6 = t6.groupby('EID').agg('mean').reset_index()
t6['RIGHT_date_min'] = pd.Series(temp1.iloc[:,1], index=t6.index)
t6['RIGHT_date_max'] = pd.Series(temp2.iloc[:,1], index=t6.index)

#8 right_year - rgyear
t7 = right[['EID','ASKDATE']]
rgyear = entbase[['EID','RGYEAR']]
t7 = pd.merge(t7,rgyear,on='EID')
t7['RIGHT_rgyear'] = t7['ASKDATE'].map(lambda x:x[:4]).astype(int) -t7['RGYEAR'].astype(int)
t7 = t7.drop(['ASKDATE','RGYEAR'],axis=1)
temp = t7.groupby('EID').agg('min').reset_index()
t7 = t7.groupby('EID').agg('max').reset_index()
t7.rename(columns={'RIGHT_rgyear':'RIGHT_rgyear_max'},inplace=True)
t7['RIGHT_rgyear_min'] = pd.Series(temp.iloc[:,1], index=t7.index)


feature_right = pd.merge(t,t1,on='EID')
feature_right = pd.merge(feature_right,t2,on = 'EID')
feature_right = pd.merge(feature_right,t3,on = 'EID')
feature_right = pd.merge(feature_right,t4,on = 'EID')
feature_right = pd.merge(feature_right,t5,on = 'EID')
feature_right = pd.merge(feature_right,t6,on = 'EID')
feature_right = pd.merge(feature_right,t7,on = 'EID')
feature_right = feature_right.drop_duplicates('EID')
feature_right.to_csv('data/feature_right.csv',index = None)
print('right succeed')

#------------------------------feature from 6project------------------------------------------
#1：project_count项目数量统计
#2: project_timegap
#3:  DJDATE_2013/2014/2015
#4: ifhome-count//ifhome_ratio
#5: last year
#6: 每年事件数发生趋势
#7: TYPECODE(最小&最大)
#9: 同项目的eid是否继续经营的和

def get_month_gap(start_month):   #输入：开始的年月，输出：离2017-08的月份差
	year,month = start_month.split('-')
	month_gaps = (2017 - int(year))*12+(8 - int(month))*1
	return month_gaps

def get_month_gap2(start_month):
	year = start_month.split('年')[0]
	month = start_month.split('年')[1].split('月')[0]
	month_gaps = (2017 - int(year))*12+(8 - int(month))*1
	return month_gaps


#1：project_count项目数量统计
t = project[['EID']]            
t['project_count'] = 1
t = t.groupby('EID').agg('sum').reset_index()

#2: project_timegap
t1 = project[['EID','DJDATE']] 
t1.DJDATE = t1.DJDATE.apply(get_month_gap)
t1 = t1.groupby('EID').agg('min').reset_index()
t1.rename(columns={'DJDATE':'project_timegap'},inplace=True)

#3:  DJDATE_2013/2014/2015
t2 = project[['EID','DJDATE']]
t2['DJDATE_Y'] = t2['DJDATE'].map(lambda x:x.split('-')[0]) 
project_DJDATE_Y = pd.get_dummies(t2['DJDATE_Y'])

t2 = pd.concat([t2.drop('DJDATE',axis = 1),project_DJDATE_Y], axis = 1)
t2 = t2.groupby('EID').agg('sum').reset_index()

#4: ifhome-count//ifhome_ratio
t3 = project[['EID','IFHOME']]
t3 = t3.groupby('EID').agg('sum').reset_index()
t3.rename(columns={'IFHOME':'project_IFHOME_count'},inplace=True)
t3 = pd.merge(t3,t,on='EID',how='left')
t3['ifhome_ratio'] = t3.project_IFHOME_count/t3.project_count
t3 = t3.drop('project_count',axis = 1)

#5: last year
t4 = project[['EID','DJDATE']]
t4['DJDATE'] = t4['DJDATE'].map(lambda x:x.split('-')[0])
t4 = t4.groupby('EID').agg('max').reset_index()
t4.rename(columns={'DJDATE':'P_year'},inplace=True)
t4.P_year = 2017 - t4.P_year.astype('int')

#6: 每年事件数发生趋势
X_parameters = np.array(project_DJDATE_Y.axes[1]).astype('int')
t2.rename(columns = {'2013':'DJDATE_2013','2014':'DJDATE_2014','2015':'DJDATE_2015'},inplace=True)
t2['project_trend'] = t2[['DJDATE_2013','DJDATE_2014','DJDATE_2015']].apply(lambda x: get_coefficient(X_parameters,x),axis = 1).iloc[:,0].round(2)

#7: TYPECODE(最小&最大)
t5 = project[['EID','TYPECODE']]
p_mintypecode = t5.sort_values('TYPECODE',ascending=True).drop_duplicates('EID')[['EID','TYPECODE']]
p_maxtypecode = t5.sort_values('TYPECODE',ascending=False).drop_duplicates('EID')[['EID','TYPECODE']]
t6 = pd.merge(p_mintypecode,p_maxtypecode,on='EID',how='left')
t6.rename(columns={'TYPECODE_x':'p_mintypecode','TYPECODE_y':'p_maxtypecode'},inplace=True)

#8:project DJDATE-rgyear
#8:project DJDATE-rgyear
t7 = project[['EID','DJDATE']]
t7['DJDATE'] = t7['DJDATE'].map(lambda x:x.split('-')[0])
rgyear = entbase[['EID','RGYEAR']]
t7 = pd.merge(t7,rgyear,on='EID')
t7['project_rgyear'] = t7['DJDATE'].astype(int) -t7['RGYEAR'].astype(int)
t7 = t7.drop(['DJDATE','RGYEAR'],axis=1)
temp = t7.groupby('EID').agg('min').reset_index()
t7 = t7.groupby('EID').agg('max').reset_index()
t7.rename(columns={'project_rgyear':'project_rgyear_max'},inplace=True)
t7['project_rgyear_min'] = pd.Series(temp.iloc[:,1], index=t7.index)

#9: 同项目的eid是否继续经营的和sametype_target_sum
table1 = project[['EID','TYPECODE']]
table2 = train[['EID','TARGET']]
MIX_table = pd.merge(table1,table2,on=['EID'],how='left')
MIX_table = MIX_table.fillna(0)
type_count = MIX_table.groupby(['TYPECODE'],as_index=False)['TARGET'].sum()   #同TYPECODE累计
type_count.rename(columns={'TARGET':'sametype_target_sum'},inplace=True)
MIX_table = pd.merge(MIX_table,type_count,on=['TYPECODE'])  #sametype_target_sum:表示同一项目ID的是否继续经营TARGET的和
MIX_table = MIX_table.groupby(['EID'],as_index=False)['sametype_target_sum'].sum()


feature_project = pd.merge(t,t1,on='EID',how='left')
feature_project = pd.merge(feature_project,t2,on='EID',how='left')
feature_project = pd.merge(feature_project,t3,on='EID',how='left')
feature_project = pd.merge(feature_project,t4,on='EID',how='left')
feature_project = pd.merge(feature_project,t5,on='EID',how='left')
feature_project = pd.merge(feature_project,t6,on='EID',how='left')
feature_project = pd.merge(feature_project,t7,on='EID',how='left').drop_duplicates('EID')
feature_project = pd.merge(feature_project,MIX_table,on='EID',how='left').drop_duplicates('EID')
feature_project.to_csv('data/feature_project.csv',index = None)
print('project succeed')
#-----------------------------feature from 7lawsuit-----------------------------------------------------
#1 month gap
#2 lawamount
#3 lawamount-zczb
#4 事件数
#5 每年被执行金额//趋势//year
#6 TYPECODE


#1 month gap
t1 = lawsuit[['EID','LAWDATE']]
t1['law_month_gap'] = t1['LAWDATE'].apply(get_month_gap2)
t1 = t1.groupby('EID',as_index=False).agg('mean')

#2 lawamount
t2 = lawsuit[['EID','LAWAMOUNT']]
t2['LAWAMOUNT'] = t2['LAWAMOUNT'].astype(int)
t2 = t2.groupby('EID',as_index=False).agg('sum')
t2['log_LAWAMOUNT'] = np.log1p(t2['LAWAMOUNT'])
t2 = t2.drop_duplicates(['EID'])

#3 lawamount-zczb
t3_1 = entbase[['EID','ZCZB']]
t3_2 = lawsuit[['EID','LAWAMOUNT']]
t3 = pd.merge(t3_2,t3_1,on=['EID'],how='left')
t3['lawamount-zczb'] = t3['LAWAMOUNT']/t3['ZCZB']
del t3['ZCZB'],t3['LAWAMOUNT']

#4 事件数
t4 = lawsuit[['EID']]
t4['law_count'] = 1
t4 = t4.groupby('EID',as_index=False).agg('sum')


#5 每年被执行金额//趋势//year
t5 = lawsuit[['EID','LAWDATE','LAWAMOUNT']]
t5['year'] = t5['LAWDATE'].map(lambda x:x.split('年')[0]).astype('int')
temp = t5.groupby(['EID','year'],as_index=False)['LAWAMOUNT'].sum()
year = pd.get_dummies(t5.year)
year_col_name = year.columns.tolist()
del t5['LAWDATE']
t5= pd.concat([t5, year], axis=1)
for name in year_col_name:
	t5[name] *= t5['LAWAMOUNT']
del t5['LAWAMOUNT']
X_parameters1 = np.array(year.axes[1]).astype('int')
t5['lamount_trend'] = t5.iloc[:,1:-1].apply(lambda x: get_coefficient(X_parameters1,x),axis = 1).iloc[:,0].round(2)
t5.year = 2017 - t5.year.astype(int)
t5 = t5.drop_duplicates('EID')

#6 TYPECODE
t6 = lawsuit[['EID','TYPECODE']]
t6 = t6.drop_duplicates(['EID'])

feature_lawsuit = pd.merge(t1,t2,on=['EID'],how='left')
feature_lawsuit = pd.merge(feature_lawsuit,t3,on='EID',how='left')
feature_lawsuit = pd.merge(feature_lawsuit,t4,on='EID',how='left')
feature_lawsuit = pd.merge(feature_lawsuit,t5,on='EID',how='left')
feature_lawsuit = pd.merge(feature_lawsuit,t6,on='EID',how='left').drop_duplicates('EID')
feature_lawsuit.to_csv('data/feature_lawsuit.csv',index = None)  
print('lawsuit succeed')
#----------------------------feature from 8breakfaith-----------------------------------------------------
#1 失信最早年份
#2 SXENDDATE
#3 每年发生次数
#4 month-gap
#5 TYPECODE



#1 失信最早年份
t1 = breakfaith[['EID','FBDATE']]
t1['FBDATE_Y'] = t1['FBDATE'].map(lambda x:x.split('年')[0])
t1 =t1.groupby('EID').agg('min').reset_index()
t1.FBDATE_Y = 2017 - t1.FBDATE_Y.astype(int)
del t1['FBDATE']

#2 SXENDDATE
t2 = breakfaith[['EID','SXENDDATE']]
t2['SXENDDATE'] = breakfaith['SXENDDATE'].fillna(0)
t2['is_breakfaith'] = (t2['SXENDDATE']!=0 ).astype('int')
del t2['SXENDDATE']
break_count = t2.groupby('EID',as_index=False)['is_breakfaith'].count()
break_sum = t2.groupby('EID',as_index=False)['is_breakfaith'].sum()
break_count.rename(columns={'is_breakfaith':'break_count'},inplace=True)
break_sum.rename(columns={'is_breakfaith':'break_sum'},inplace=True)
t2 = pd.merge(t2,break_count,on='EID',how='left')
t2 = pd.merge(t2,break_sum,on='EID',how='left')


#3 每年发生次数
t3 = breakfaith[['EID','FBDATE']]
t3['FBDATE_Y'] = t3['FBDATE'].map(lambda x:x.split('年')[0])
breakfaith_year = pd.get_dummies(t3['FBDATE_Y'] ,prefix='breakfaithYEAR') 
breakfaith_year1 = pd.get_dummies(t3['FBDATE_Y']) 
t3 = pd.concat([t3['EID'],breakfaith_year],axis=1) 
t3 = t3.groupby('EID',as_index=False).sum()
X_parameters = np.array(breakfaith_year1.axes[1]).astype('int')
t3['breakfaith_trend'] = t3[['breakfaithYEAR_2013','breakfaithYEAR_2014','breakfaithYEAR_2015']].apply(lambda x: get_coefficient(X_parameters,x),axis = 1).iloc[:,0].round(2)

#4 month-gap
t4 = breakfaith[['EID','FBDATE','SXENDDATE']]
t4['SXENDDATE'] = t4['SXENDDATE'].fillna('201708')
t4['FBDATE'] = t4['FBDATE'].apply(lambda x: re.sub(r'\D', '', str(x)))
t4['FBDATE'] = t4['FBDATE'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m'))
t4['SXENDDATE'] = t4['SXENDDATE'].apply(lambda x: re.sub(r'\D', '', str(x)))
t4['SXENDDATE'] = t4['SXENDDATE'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m'))
t4['break_month_gap'] = ((t4['SXENDDATE']-t4['FBDATE'])/30).apply(lambda x:re.findall(r'\d+',str(x))[0])
t4['break_month_gap'] = t4['break_month_gap'].astype('int')
del t4['FBDATE'],t4['SXENDDATE']
t4 = t4.drop_duplicates('EID')

#5 TYPECODE
t5 = breakfaith[['EID','TYPECODE']]

#7 breakfaith - rgyear
t7 = breakfaith[['EID','FBDATE']].copy()
t7['FBDATE'] = t7['FBDATE'].map(lambda x:x.split('年')[0])
rgyear = entbase[['EID','RGYEAR']]
t7 = pd.merge(t7,rgyear,on='EID')
t7['breakf_rgyear'] = t7['FBDATE'].astype(int) -t7['RGYEAR'].astype(int)
t7 = t7.drop(['FBDATE','RGYEAR'],axis=1)
temp = t7.groupby('EID').agg('min').reset_index()
t7 = t7.groupby('EID').agg('max').reset_index()
t7.rename(columns={'breakf_rgyear':'breakf_rgyear_max'},inplace=True)
t7['breakf_rgyear_min'] = pd.Series(temp.iloc[:,1], index=t7.index)


feature_breakfaith = pd.merge(t1,t2,on='EID',how='left')
feature_breakfaith = pd.merge(feature_breakfaith,t3,on='EID',how='left')
feature_breakfaith = pd.merge(feature_breakfaith,t4,on='EID',how='left')
feature_breakfaith = pd.merge(feature_breakfaith,t5,on='EID',how='left')
feature_breakfaith = pd.merge(feature_breakfaith,t7,on='EID',how='left').drop_duplicates('EID')
feature_breakfaith['break_ratio'] = feature_breakfaith['break_sum'] / feature_breakfaith['break_count']  
feature_breakfaith.to_csv('data/feature_breakfaith.csv',index = None)
print('breakfaith succeed')

#---------------------------feature from 9recruit-------------------------------------------------------
#特证1：recruit_RECRNUM_count 招聘总次数
#特证2：recruit_RECRNUM_sum 招聘职位总数[重要]
#特征3：recurt_info_ration平均每次招聘的职位数量[重要]
#特征4：WZCODE_ZP01~ZP03:招聘WZCODE的one-hot  
#特征5：month_gap:最近招聘的月度gap
#1  recuit_year
t1 = recruit[['EID','RECDATE']] 
t1['RECDATE'] = t1['RECDATE'].map(lambda x:x.split('-')[0])
t1 = t1.groupby('EID').agg('max').reset_index()
t1.rename(columns={'RECDATE':'recruit_year'},inplace=True)
t1.recruit_year =2017 - t1.recruit_year.astype('int')

#2 WZCODE
t2 = recruit[['EID','WZCODE']] 
recruit_WZCODE = pd.get_dummies(t2['WZCODE'],prefix='WZCODE')
t2 = pd.concat([t2['EID'],recruit_WZCODE],axis=1)
t2 = t2.groupby('EID').agg('sum').reset_index()


#3 recuit_count
t3 = recruit[['EID']]            
t3['recuit_count'] = 1
t3 = t3.groupby('EID').agg('sum').reset_index()

#4 r_month_gap
t4 = recruit[['EID','RECDATE']] 
t4['r_month_gap'] = t4['RECDATE'].apply(get_month_gap)
t4 = t4.groupby('EID').agg('max').reset_index()
t4 = t4.drop('RECDATE',axis = 1)

#5 postcode
#t5 = recruit[['EID','POSCODE']] 

#6 PNUM
t6 = recruit[['EID','PNUM']] 
t6 = t6.fillna('若干')
t6['PNUM'] = t6['PNUM'].apply(lambda x:re.sub(r'\D','', str(x)))
t6['PNUM'] = t6['PNUM'].apply(lambda x:0 if x=='' else x)
t6['PNUM'] = t6['PNUM'].astype(int)
t6.to_csv('data/t6_0.csv',index=None)
mean = t6.PNUM.mean()
t6['PNUM'] = t6['PNUM'].apply(lambda x:mean if x==0 else x)
t6.to_csv('data/t6_1.csv',index=None)
t7 = t6.copy()
t6 = t6.groupby('EID').agg('mean').reset_index()
t6['PNUM'] = t6['PNUM'].round(2)
t6.to_csv('data/t6_2.csv',index=None)
#7 SUM_PNUM
t7 = t7.groupby('EID').agg('sum').reset_index()
t7.rename(columns={'PNUM':'SUM_PNUM'},inplace=True)
t7['SUM_PNUM'] = t7['SUM_PNUM'].round(2)
t7.to_csv('data/t7.csv',index=None)

#8 recruit - rgyear
t8 = recruit[['EID','RECDATE']].copy()
t8['RECDATE'] = t8['RECDATE'].map(lambda x:x.split('-')[0])
rgyear = entbase[['EID','RGYEAR']]
t8 = pd.merge(t8,rgyear,on='EID')
t8['recuit_rgyear'] = t8['RECDATE'].astype(int) -t8['RGYEAR'].astype(int)
t8 = t8.drop(['RECDATE','RGYEAR'],axis=1)
temp = t8.groupby('EID').agg('min').reset_index()
t8 = t8.groupby('EID').agg('max').reset_index()
t8.rename(columns={'recuit_rgyear':'recuit_rgyear_max'},inplace=True)
t8['recuit_rgyear_min'] = pd.Series(temp.iloc[:,1], index=t8.index)

feature_recruit = pd.merge(t1,t2,on=['EID'],how='left')
feature_recruit = pd.merge(feature_recruit,t3,on='EID',how='left')
feature_recruit = pd.merge(feature_recruit,t4,on='EID',how='left')
#feature_recruit = pd.merge(feature_recruit,t5,on='EID',how='left')
feature_recruit = pd.merge(feature_recruit,t6,on='EID',how='left')
feature_recruit = pd.merge(feature_recruit,t7,on='EID',how='left')
feature_recruit = pd.merge(feature_recruit,t8,on='EID',how='left').drop_duplicates('EID')
feature_recruit.to_csv('data/feature_recruit.csv',index = None)
print('recruit succeed')
#---------------------------feature from 10qualification-------------------------------------------------------
#1 addtype
#2 资质数量
#3 



#1 addtype
t1 = qualification[['EID','ADDTYPE']] 
ADDTYPE = pd.get_dummies(t1['ADDTYPE'],prefix='ADDTYPE')
t1 = pd.concat([t1['EID'],ADDTYPE],axis=1)
t1 = t1.groupby('EID').agg('sum').reset_index()

#2 资质数量
t2 = qualification[['EID']]
t2['quali_count'] = 1
t2 = t2.groupby('EID').agg('sum').reset_index()

#3 
t3 = qualification[['EID','BEGINDATE','EXPIRYDATE']]
t3 = t3.dropna(axis=0,how='any')
t3['BEGINDATE'] = t3['BEGINDATE'].apply(lambda x: re.sub(r'\D', '', str(x)))
t3['BEGINDATE'] = t3['BEGINDATE'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m'))
t3['EXPIRYDATE'] = t3['EXPIRYDATE'].apply(lambda x: re.sub(r'\D', '', str(x)))
t3['EXPIRYDATE'] = t3['EXPIRYDATE'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m'))
t3['qua_month_gap'] = ((t3['EXPIRYDATE']-t3['BEGINDATE'])/30).apply(lambda x:re.findall(r'\d+',str(x))[0])
t3['qua_month_gap'] = t3['qua_month_gap'].astype('int')
del t3['EXPIRYDATE'],t3['BEGINDATE']
t3 = t3.groupby('EID').agg('mean').reset_index()

feature_qualification = pd.merge(t1,t2,on=['EID'],how='left')
feature_qualification = pd.merge(feature_qualification,t3,on=['EID'],how='left').drop_duplicates(['EID'])
feature_qualification.to_csv('data/feature_qualification.csv',index = None)
print('qualification succeed')

######另外增加的特征






feature_all = pd.merge(feature_entbase,feature_alter,on = 'EID', how = 'left')
feature_all = pd.merge(feature_all,feature_branch,on = 'EID', how = 'left')
feature_all = pd.merge(feature_all,feature_invest,on = 'EID', how = 'left')
feature_all = pd.merge(feature_all,feature_right,on = 'EID', how = 'left')
feature_all = pd.merge(feature_all,feature_project,on = 'EID', how = 'left')
feature_all = pd.merge(feature_all,feature_lawsuit,on = 'EID', how = 'left')
feature_all = pd.merge(feature_all,feature_breakfaith,on = 'EID', how = 'left')
feature_all = pd.merge(feature_all,feature_recruit,on = 'EID', how = 'left')
feature_all = pd.merge(feature_all,feature_qualification,on = 'EID', how = 'left')
feature_all = pd.merge(feature_all,t1,on = 'EID', how = 'left')

feature_all = feature_all.fillna(-1)
feature_all = feature_all.drop_duplicates(['EID'])

feature_all['EID'] = list(map(peid2eid, list(feature_all['EID'])))
feature_all.to_csv('data/feature_all.csv',index = None)
print('--------feature_all succeed-------------')


train['EID'] = list(map(peid2eid, list(train['EID'])))
evaluation_public['EID'] = list(map(peid2eid, list(evaluation_public['EID'])))
feature_train = pd.merge(feature_all, train, on = 'EID', how = 'right')
feature_evaluation = pd.merge(feature_all, evaluation_public, on = 'EID', how = 'right')



#feature_train.dtypes.to_csv('data/dtypes.csv',index = None)
feature_train.to_csv('data/feature_train.csv',index = None)
feature_evaluation.to_csv('data/feature_evaluation.csv',index = None)


print(train.TARGET.astype('int').sum())
print ('feature extract succed!')
#结果为29092，表示所有的测试数据中停业的为29092个，总数为153006个

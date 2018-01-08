#coding=utf-8
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

def eid2peid(x):
    if x < 600000:
        return 'p' + str(x)
    else:
        return 's' + str(x - 600000)
#提取训练数据集
dataset = pd.read_csv('data/feature_train.csv')
dataset.fillna(-1)
droplist = ['EID','TARGET','ENDDATE','MPNUM+ENUM+FSTINUM+TZINUM']
dataset_x = dataset.drop(droplist,axis=1)
dataset_y = dataset.TARGET


#将训练数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.2)


#提取预测数据集
X_pred = pd.read_csv('data/feature_evaluation.csv')
X_pred = X_pred.fillna(-1)
evaluation_index = X_pred.EID

droplist2 = ['EID','MPNUM+ENUM+FSTINUM+TZINUM']
X_pred = X_pred.drop(droplist2,axis=1)


#####################  Xgboost  ##################################

train_set = xgb.DMatrix(X_train, label = y_train)
test_set = xgb.DMatrix(X_test, label = y_test)
evaluation_set = xgb.DMatrix(X_pred)


params={
		'booster':'gbtree',
	    'objective': 'binary:logistic',
	    'eval_metric':'auc',
	    #'learning_rate':0.02,
	    'gamma':0.1,   #0.1
	    'min_child_weight':2.1,   #1.1 调大控制过拟合
	    'max_depth':8,
	    'lambda':10,    #用来降低过拟合
	    'subsample':0.7,    
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'n_estimators':750,  #228
	    'eta': 0.01,   #缩减权重防止过拟合
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12

	    }

model = xgb.train(params,train_set,num_boost_round = 3000)

#predict test set
y_test_xgb = model.predict(test_set)

#predict evaluation set
y_evaluation_xgb = model.predict(evaluation_set)
np.savetxt('data/y_evaluation_xgb.csv',y_evaluation_xgb,delimiter = ',')
np.savetxt('data/y_test_xgb.csv',y_test_xgb,delimiter = ',')

y_evaluation_xgb = np.round(y_evaluation_xgb,8)
result = pd.DataFrame({'PROB':list(y_evaluation_xgb),
                       })
result['FORTARGET'] = result['PROB'] > 0.22
result['PROB'] = result['PROB'].astype('str')
result['FORTARGET'] = result['FORTARGET'].astype('int')

result = pd.concat([evaluation_index,result],axis=1)

result[['EID','FORTARGET','PROB']].to_csv('Result/evaluation_public_xgb.csv',index=None)



feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key,value) in feature_score:
    fs.append("{0},{1}\n".format(key,value))
    
with open('xgb_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)


################################  lgb  #####################################
print('--------start--------')
train_test_set = lgb.Dataset(dataset_x,dataset_y)
train_set = lgb.Dataset(X_train, y_train)
test_set = lgb.Dataset(X_test, y_test, reference = train_set)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'n_estimators': 2000,
    'lambda_l1':3.0,
    'lambda_l2':3.0,
    'num_leaves': 128,
    'learning_rate': 0.005,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.9,    
    'min_data_in_leaf':20,
    'min_child_weight':0.01,
    'verbose': 0,
    'n_jobs': 16,
}



evals_result = {}

gbm = lgb.train(params, 
				train_set, 
				num_boost_round = 500, 
				valid_sets=test_set, 
				early_stopping_rounds= 100,
				evals_result=evals_result
				)

# print('Plot metrics during training...')
# ax = lgb.plot_metric(evals_result, metric='auc')
# plt.show()

# print('Plot feature importances...')
# lgb.plot_importance(gbm,max_num_features=X_train.shape[1])
# plt.show()

y_test_lgb = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_evaluation_lgb = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
y_evaluation_lgb  = np.round(y_evaluation_lgb ,8)


##############################  DART #####################################

params = {
    'boosting_type': 'dart',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 128,
    'learning_rate': 0.1,  #0.08
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 10,
    'verbose': 0
}
evals_result = {}

gbm = lgb.train(params, 
				train_set, 
				num_boost_round = 3000, 
				valid_sets=test_set, 
				early_stopping_rounds= 100,
				evals_result=evals_result
				)

y_test_lgb2 = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_evaluation_lgb2 = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
y_evaluation_lgb2  = np.round(y_evaluation_lgb2 ,8)



print('xgb score:',roc_auc_score(y_test, y_test_xgb))
print('lgb_gbdt score:',roc_auc_score(y_test, y_test_lgb))
print('lgb_DART score:',roc_auc_score(y_test, y_test_lgb2))
print(roc_auc_score(y_test, (y_test_lgb+y_test_xgb+y_test_lgb2)/3))

#lgb-gdbt
y_evaluation_lgb = np.round(y_evaluation_lgb,8)
result = pd.DataFrame({'PROB':list(y_evaluation_lgb),
                       })
result['FORTARGET'] = result['PROB'] > 0.22
result['PROB'] = result['PROB'].astype('str')
result['FORTARGET'] = result['FORTARGET'].astype('int')

result = pd.concat([evaluation_index,result],axis=1)
result[['EID','FORTARGET','PROB']].to_csv('Result/evaluation_public_lgb_gdbt.csv',index=None)


#lgb-dart
y_evaluation_lgb2 = np.round(y_evaluation_lgb2,8)
result = pd.DataFrame({'PROB':list(y_evaluation_lgb2),
                       })
result['FORTARGET'] = result['PROB'] > 0.22
result['PROB'] = result['PROB'].astype('str')
result['FORTARGET'] = result['FORTARGET'].astype('int')

result = pd.concat([evaluation_index,result],axis=1)
result[['EID','FORTARGET','PROB']].to_csv('Result/evaluation_public_lgb_dart.csv',index=None)



###############################   ensembling    ############################### 
y_evaluation_ensembling = (y_evaluation_xgb + y_evaluation_lgb + y_evaluation_lgb2 )/3
result = pd.DataFrame({'PROB':list(y_evaluation_ensembling),
                       })
result['FORTARGET'] = result['PROB'] > 0.22
result['PROB'] = result['PROB'].astype('str')
result['FORTARGET'] = result['FORTARGET'].astype('int')

result = pd.concat([evaluation_index,result],axis=1)
result[['EID','FORTARGET','PROB']].to_csv('Result/evaluation_public.csv',index=None)
print ('succeed-------')
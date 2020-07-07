# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd
import seaborn as sns
%matplotlib inline

d=pd.read_excel('dataset.xlsx')


dataset_rd=d.iloc[list((d.iloc[:,6:].dropna(how = 'all')).index),:]
blood_d=dataset_rd.iloc[:,7:21]


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(blood_d)
blood_d_im=imp.transform(blood_d)
blood_d_im= pd.DataFrame(blood_d_im, columns=blood_d.columns)

figure=sns.heatmap(blood_d_im)
figure.figure.savefig('blood_hm.png', vmin=0, vmax=1,cmap="YlGnBu", dpi=2000)

cate_d=dataset_rd.dropna(axis=1,thresh=1000).iloc[:,7:]
cate_d.fillna(2, inplace = True)
cate_d=cate_d.replace({'not_detected': 0,'detected':1})
cate_d=cate_d.reset_index(drop=True)
figure=sns.heatmap(cate_d)
figure.figure.savefig('categ_hm.png', vmin=0, vmax=1,cmap="YlGnBu", dpi=2000)

(set((dataset_rd.dropna(axis=1,thresh=100).columns)) -  set((dataset_rd.dropna(axis=1,thresh=1000).columns)))


x=pd.concat([dataset_rd.iloc[:,[1,2]].reset_index(drop=True),blood_d_im,cate_d ], axis=1, sort=False)
y=dataset_rd.iloc[:,[3,4,5]]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
x.iloc[:,1]=le.fit_transform(x.iloc[:,1])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state = 0)
classifier.fit(x_train,y_train)

classifier.score(x_test,y_test)

y_pred = classifier.predict(x_test)


import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
cmm = multilabel_confusion_matrix(y_test, y_pred)


from sklearn.tree import DecisionTreeClassifier
classifierdt = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
classifierdt.fit(x_train,y_train)

classifierdt.score(x_test,y_test)


import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
cmdt = multilabel_confusion_matrix(y_test, y_pred)
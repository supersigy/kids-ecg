"""
Created on Mon May 03 13:24:22 2021
@author: Gabor Asztalos

input: all patient csv with demog data and GEMuse signal output
output: random forest model to predict future rhythm classification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


#load dataset containing muse calculated ecg features, demog data, and labels
dataset = pd.read_csv(r'C:\Users\super\Documents\Learning\EKG\master.csv')

#clean categroical data
categorical_columns = ['GENDER','RACE']
for cc in categorical_columns:
    dataset[cc].fillna('NA', inplace=True)
cat_data = dataset[categorical_columns]
#clean numeric data
numeric_columns = ['Age','ATRIALRATE','P_RINTERVAL','QRSDURATION','Q_TINTERVAL',
                   'QTCCALCULATION','PAXIS','RAXIS','TAXIS','QRSCOUNT', 
                   'QONSET','QOFFSET','PONSET','POFFSET','TOFFSET']
for nc in numeric_columns:
    dataset[nc].fillna(0, inplace=True)

sc = StandardScaler()
dataset[numeric_columns] = sc.fit_transform(dataset[numeric_columns])

#set target column and find output size    
target_columns = ['rhythm_label']
dataset.loc[dataset['rhythm_label']==99, 'rhythm_label'] = 31 

#encode categorical data
for c in categorical_columns:
    dataset[c] = dataset[c].astype('category')
dataset['GENDER'] = dataset['GENDER'].cat.codes.values
dataset['RACE'] = dataset['RACE'].cat.codes.values

#turn features and labels into X,targets tensor
features = categorical_columns + numeric_columns
X = dataset[features]
targets = dataset[target_columns].values.flatten()

#test,train split of dataset
X_train, X_test, y_train, y_test = train_test_split(X, targets, 
                                    test_size=0.2,random_state=199)

model = RandomForestClassifier()
model.fit(X_train,y_train)

model.score(X_test,y_test)

y_preds = model.predict(X_test)

print(confusion_matrix(y_test,y_preds))
print(classification_report(y_test,y_preds))
print(accuracy_score(y_test, y_preds))


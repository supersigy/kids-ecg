"""
Created on Fri Apr 09 08:42:12 2021
@author: Gabor Asztalos

input: master list of all ECGs with demographic(categorical) and GE MUSE features
output: cleaned up master list with target labels encoded by Dx1-Dx4 language logic
"""


import pandas as pd
import numpy as np
import os
from nltk.stem import WordNetLemmatizer
import re

def clean_data(dx):
    if str(dx) == 'nan':
        dx = ''
    stemmer = WordNetLemmatizer()
    dx = dx.lower()
    dx = dx.replace('-','')
    dx = dx.replace('1st', 'first')
    dx = dx.replace('2nd', 'second')
    dx = dx.replace('3rd', 'third')
    dx = dx.replace('4th', 'fourth')
    dx = re.sub(r'\W', ' ', dx)
    dx = re.sub(r'\s+', ' ', dx, flags=re.I)
    tokens = dx.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    dx = ' '.join(tokens)
    if 'referring' in(dx):
        dx = ''
    if 'pediatric ecg analysis' in(dx):
        dx = ''
    if 'edited' in(dx):
        dx = ''
    return dx

def rhythm_labeler(dx):
    label = 0
    if 'normal' in(dx) or 'sinus rhythm' in(dx) or 'sinus arrhythmia' in(dx):
        label = 1
    if 'sinus tachycardia' in(dx):
        label = 2
    if 'sinus bradycardia' in(dx):
        label = 3
    if 'ectopic atrial rhythm' in(dx) or 'ectopic atrial bradycardia' in(dx) or 'atrial rhythm' in(dx):
        label = 4
    if 'ectopic atrial tachycardia' in(dx) or re.search(r'\beat\b', dx):
        label = 5
    if 'pacemaker' in(dx) or 'wap' in(dx):
        label = 6 
    if 'low right atrial rhythm' in(dx):
        label = 7
    if 'atrial flutter' in(dx) or 'a flutter' in(dx):
        label = 8
    if 'atrial fibrillation' in(dx) or 'a fib' in(dx) or 'afib' in(dx):
        label = 9
    if 'avrt' in(dx) or 'avnrt' in(dx) or 'reentrant' in(dx):
        label = 10
    if 'junctional tachycardia' in(dx) or 'pjrt' in(dx):
        label = 11
    if 'supraventricular tachycardia' in(dx):
        label = 12
    if ('atrial tachycardia' in(dx) or 'atach' in(dx)) and 'ectopic atrial tach' not in(dx):
        label = 13   
    if 'idioventricular rhythm' in(dx):
        label = 14
    if ('ventricular tachycardia' in(dx) or 'morphic' in(dx) or re.search(r'\bvt\b', dx)) and 'supraventricular tachycardia' not in(dx) :
        label = 15
    if 'ventricular fibrillation' in(dx) or 'vfib' in(dx) or 'v fib' in(dx):
        label = 16
    if 'junctional rhythm' in(dx):
        label = 99
    if 'accelerated' in(dx):
        label = 17
    if 'junctional ectopic tachycardia' in(dx) or 'jet' in(dx):
        label = 18
    if 'junctional bradycardia' in(dx):
        label = 19
    if 'first degree' in(dx):
        label = 20
    if 'second degree' in(dx) or 'mobitz' in(dx):
        label = 21
    if 'third degree' in(dx) or 'complete heart block' in(dx):
        label = 22
    if 'premature atrial complex' in(dx) or re.search(r'\bpac\b', dx):
        label = 23
    if 'premature ventricular complex' in(dx) or 'pvc' in(dx):
        label = 24
    if 'premature junctional complex' in(dx) or 'pjc' in(dx):
        label = 25
    if 'right bundle branch block' in(dx) or'rbbb' in(dx):
        label = 26
    if 'left bundle branch block' in(dx) or'lbbb' in(dx):
        label = 27 
    if 'hemiblock' in(dx) or 'fascicular' in(dx):
        label = 28
    if 'conduction delay' in(dx) or 'ivcd' in(dx):
        label = 29
    if 'paced rhythm' in(dx):
        label = 30
    return label

def hypertrophy_labeler(dx):
    label = 0
    if 'right atrial enlargement' in(dx) or re.search(r'\brae\b', dx):
        label = 31
    if 'left atrial enlargement' in(dx) or re.search(r'\blae\b', dx):
        label = 32
    if 'right ventricular hypertrophy' in(dx) or 'rvh' in(dx):
        label = 33
    if 'left ventricular hypertrophy' in(dx) or 'lvh' in(dx):
        label = 34   
    return label 

def morphology_labeler(dx):
    label = 0
    if 'right axis deviation' in(dx) or re.search(r'\brad\b', dx):
        label = 35
    if 'left axis deviation' in(dx) or re.search(r'\blad\b', dx):
        label = 36
    if 'northwest axis' in(dx):
        label = 37
    if 'preexcitation' in(dx):
        label = 38
    if 'brugada' in(dx):
        label = 39
    if 'dextrocardia' in(dx):
        label = 40
    if 'low voltage' in(dx):
        label = 41
    if 'st elevation' in(dx):
        label = 42
    if 'st depression' in(dx) or 'st segment depression' in(dx):
        label = 43
    if 'strain pattern' in(dx):
        label = 44
    if 'deep q wave' in(dx):
        label = 45
    if 'pr depression' in(dx):
        label = 46   
    if 't wave inversion' in(dx):
        label = 47
    if 't wave abnormality' in(dx):
        label = 48
    if 'segment abnormality' in(dx):
        label = 49
    return label

def longqt_labeler(dx):
    label = 0
    if 'prolonged qtc' in(dx):
        label = 50
    if 'borderline prolonged' in(dx) or 'boderline qtc' in(dx):
        label = 51
    return label

def poordata_labeler(dx):
    label = 0
    if 'poor data quality' in(dx) or 'incomplete study' in(dx) or 'waveform not' in(dx):
        label = 52
    if 'suspect' in(dx) or 'lead reversal' in(dx) or 'limb lead' in(dx):
        label = 53
    return label

os.chdir(r'C:\Users\super\Documents\Learning\EKG')
df = pd.read_excel('Lookup_Master 2010-2021.xlsx', engine='openpyxl')

dx_cols = ['DX'+str(i) for i in range(1,5)]
df2 = df[dx_cols]

df2[dx_cols] = df2[dx_cols].applymap(clean_data)
df2['DXX'] = df2['DX1'] + ' ' + df2['DX2'] + ' ' + df2['DX3'] + ' ' + df2['DX4']
df2['DXX'] = df2['DXX'].str.strip()
df2['DXX'] = df2['DXX'].str.replace('  ',' ')

df2['rhythm_label'] = df2['DXX'].apply(rhythm_labeler)
df2['hyper_label'] = df2['DXX'].apply(hypertrophy_labeler)
df2['morph_label'] = df2['DXX'].apply(morphology_labeler)
df2['longqt_label'] = df2['DXX'].apply(longqt_labeler)
df2['poor_label'] = df2['DXX'].apply(poordata_labeler)
df2.drop(df2.columns[[0,1,2,3]], axis=1, inplace=True)

df2.to_csv('labeler_split.csv')

dropcols = [0, 3] + list(range(7,21)) + list(range(36,79))
df.drop(df.columns[dropcols], axis = 1, inplace = True)
df3 = pd.concat([df, df2], axis=1)
df3['Age'] = ((df3['ACQUISITIONDATETIME_DT'] - df3['DOB'])/np.timedelta64(1, 'M')).astype(int)
df3['DT'] = df3['ACQUISITIONDATETIME_DT'].astype(str)
df3['DT'] = df3['DT'].str.replace('-','').str.replace(':','').str.replace(' ','')
df3['ofile'] = df3['PATIENTID'].astype(str) + '-' + df3['DT'] 
df3.drop(df3.columns[[0,1,2,27]], axis=1, inplace=True)
df3 = df3[['ofile','Age','GENDER','RACE','ATRIALRATE','P_RINTERVAL','QRSDURATION',
           'Q_TINTERVAL','QTCCALCULATION','PAXIS','RAXIS','TAXIS','QRSCOUNT',
           'QONSET','QOFFSET','PONSET','POFFSET','TOFFSET','DXX','rhythm_label',
           'hyper_label','morph_label','longqt_label','poor_label']]
df4 = df3[['ofile','rhythm_label']]
df3.to_csv('master.csv')
df4.to_csv('ecg_rhythm_dataloader.csv')
    

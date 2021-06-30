"""
Created on Wed Apr 07 08:34:40 2021
@author: Gabor Asztalos

input: master file of all patient demographics and diagnosis info for that ECG reading
output: bucketed categories for Dx entries using Kmeans on text in Dx fields
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
import os
from sklearn.metrics.pairwise import paired_distances
#import nltk
#nltk.download('wordnet')
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#from nlkt.corpus import wordnet

#function to clean up Dx data, handle missings, lemmatize to word stems
def clean_data(row_dx):
    stemmer = WordNetLemmatizer()
    for i in range(0, len(row_dx)):
        row_dx[i] = row_dx[i].lower()
        row_dx[i] = row_dx[i].replace('-','')
        row_dx[i] = row_dx[i].replace('1st', 'first')
        row_dx[i] = row_dx[i].replace('2nd', 'second')
        row_dx[i] = row_dx[i].replace('3rd', 'third')
        row_dx[i] = row_dx[i].replace('4th', 'fourth')
        row_dx[i] = row_dx[i].replace('1', 'one')
        row_dx[i] = row_dx[i].replace('2', 'two')
        row_dx[i] = row_dx[i].replace('type i', 'mobitz')
        row_dx[i] = row_dx[i].replace('type ii', 'mobitz')
        row_dx[i] = re.sub(r'\W', ' ', row_dx[i])
        row_dx[i] = re.sub(r'\s+', ' ', row_dx[i], flags=re.I)
        tokens = row_dx[i].split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        row_dx[i] = ' '.join(tokens)
        if 'with' in(row_dx[i]):
            w = re.split(' with ', row_dx[i], maxsplit=3)
            row_dx[i] = w[0]
    return row_dx

os.chdir(r'C:\Users\super\Documents\Learning\EKG')

df = pd.read_excel('Lookup_Master 2010-2021.xlsx', engine='openpyxl')
labels_df = pd.read_csv('labels_vB.csv', header=0)
labels = labels_df['Label'].to_numpy()
labels = clean_data(labels)

dx_cols = ['DX'+str(i) for i in range(1,3)]

#set UNK type Dx labels to -1 and run clean data func
dx_matrix = df[dx_cols]
mask = (dx_matrix['DX1'].str.find('PEDIATRIC ECG ANALYSIS') != -1)
dx_matrix['DX1'][mask] = dx_matrix['DX2']
mask2 = (dx_matrix['DX1'].str.find('INCOMPLETE STUDY') != -1)
mask3 = (dx_matrix['DX1'].str.find('WAVEFORM NOT IN') != -1)
dx_matrix['DX1'] = dx_matrix['DX1'].fillna('normal sinus rhythm')
dx_matrix['DX1'][mask2] = 'poor data quality'
dx_matrix['DX1'][mask3] = 'poor data quality'
row_dx = dx_matrix['DX1'].to_numpy()
row_dx = clean_data(row_dx)
row_dx = np.append(row_dx, labels)

#vectorize Dx text to numeric    
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(row_dx).toarray()
centers = X[len(X)-len(labels):len(X)]

model = KMeans(n_clusters=32, init='k-means++', max_iter=500, n_init=50)
#model = KMeans(n_clusters = len(labels), init = centers, n_init=1)
model.fit(X)

#append our derived cluster to dataset
predicted = model.predict(X)
closest = np.array([model.cluster_centers_[c-1] for c in predicted])
euc = paired_distances(X, closest)
dx = pd.Series(row_dx)
predicted = pd.Series(predicted)
euc = pd.Series(euc)

frame = {'DX': dx, 'K': predicted, 'EucDist': euc}
df2 = pd.DataFrame(frame)

df2.to_excel('kmeans32b.xlsx')






    

    







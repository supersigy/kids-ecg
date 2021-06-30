"""
Created on Thur Apr 29 20:12:45 2021
@author: Gabor Asztalos

input: directory poiting to large files of 5000+ patient ECG waveforms
output: output directory filed with csv files and csv-to-target file for
        pytorch dataloader
"""


import os
import pandas as pd

inpath = r'C:\Users\super\Documents\Learning\EKG\ECG-Data\Chopform'
outpath = r'C:\Users\super\Documents\Learning\EKG\ECG-Data\DataLoad'
outpath_txt = 'C:\\Users\\super\\Documents\\Learning\\EKG\\ECG-Data\\DataLoad\\'

os.chdir(inpath)

#function reads large file and takes 11 rows(for 11 leads for patient) and
#saves into smaller file
def one_file(f,num):
    smallfile = None
    with open(f) as bigfile:
        next(bigfile)
        for lineno, line in enumerate(bigfile):
            if lineno % leads == 0:
                if smallfile:
                    smallfile.close()
                small_filename = outpath+f'\single_ecg_{num}_{lineno+leads}.csv'
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

#convert all ECG large files(5000+ patients) to individual files
num = 0
leads = 11
files = os.listdir()
for f in files:
    one_file(f,num)
    num += 1

#list to remove small.txt once converted to csv    
os.chdir(outpath)
files = os.listdir()
todelete = []

#ouput directory filled with csv files w/ size 11x2500, name=patientid+date
for f in files:
    df = pd.read_csv(f, header=None)
    patientid = df.iloc[1][1]
    patientid = str(patientid)
    datetime = df.iloc[1][2]
    datetime = str.replace(datetime, '-','')
    datetime = str.replace(datetime, ':','')
    datetime = str.replace(datetime, ' ','')
    ofile_name = patientid + '-' + datetime + '.csv'
    #df.drop(df.columns[dropcols], axis = 1, inplace = True)
    df = df.iloc[:,11:2511]
    df.to_csv(ofile_name,header=False, index=False)
    todelete.append(f)

#delete txt files after csv creation    
for d in todelete:
    os.remove(d)

#because there are 160k+ ECGs in ~25 files do a random spot across all files
#to see if any are missing and need to investigate i/o 
check = []
for i in range(11,28):
    df = pd.read_csv('single_ecg_'+str(i)+'_20757.csv', header=None)
    patientid = df.iloc[1][1]
    patientid = str(patientid)
    datetime = df.iloc[1][2]
    datetime = str.replace(datetime, '-','')
    datetime = str.replace(datetime, ':','')
    datetime = str.replace(datetime, ' ','')
    ofile_name = patientid + '-' + datetime + '.csv'
    check.append(ofile_name)
    df = pd.read_csv('single_ecg_'+str(i)+'_18084.csv', header=None)
    patientid = df.iloc[1][1]
    patientid = str(patientid)
    datetime = df.iloc[1][2]
    datetime = str.replace(datetime, '-','')
    datetime = str.replace(datetime, ':','')
    datetime = str.replace(datetime, ' ','')
    ofile_name = patientid + '-' + datetime + '.csv'
    check.append(ofile_name)
    
os.chdir(r'C:\Users\super\Desktop\temp')
files = os.listdir()

#missing list will be empty after this code if all checks successful
missing = []
not_missing = 0
num = 0
missing_idx = []
for c in check:
    if c in(files):
        not_missing += 1
    else: 
        missing.append(c)
        missing_idx.append(num)
    num += 1
    
#create csv with filename for ecg in load directory coupled to target value
#here we will be trying to predict QTC on each ECG
dataset = pd.read_csv(r'C:\Users\super\Documents\Learning\EKG\master.csv')
targets = dataset[['ofile','QTCCALCULATION']]
targets['ofile'] = targets['ofile']+'.csv'
os.chdir(r'C:\Users\super\Documents\Learning\EKG\ECG-Data\DataLoad')
files =  os.listdir()
files_df = pd.DataFrame(files,columns=['ofile'])
output = pd.merge(targets, files_df, on='ofile')
output['ofile'] = outpath_txt + output['ofile']
output.to_csv(r'C:\Users\super\Documents\Learning\EKG\file_to_target.csv',index=False)


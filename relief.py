#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import sys

#%%
def readWaterMelon():
    df=pd.read_csv('watermelon3.csv',delimiter=',')
    df.iloc[:,-1] = pd.Categorical(df.iloc[:,-1])
    df['code'] = df.iloc[:,-1].cat.codes
    df['code']=df['code'].astype(str)
    df=df.drop([df.columns[0],df.columns[-2]],axis=1)
    
    df_num = df.select_dtypes(include=['float64'])
    df_norm = (df_num - df_num.min()) / (df_num.max() - df_num.min())
    df[df_norm.columns] = df_norm

    dsct=[]
    ctns=[]
    for col in df.columns[:-1]:
        if df[col].dtype=='object':
            dsct.append(col)
        else:
            ctns.append(col)
    
    return df,dsct,ctns


#%%
# sample is a data sample, i.e., a row in dataframe
def near(df,dsct,ctns,sample):
    nearDis=float('inf')
    target=None

    for idx,row in df.iterrows():
        tempDis=0.0
        for col in df.columns[:-1]:
            if col in dsct:
                tempDis+=float(row[col]==sample[col])
            else:
                tempDis+=abs(row[col]-sample[col])
        
        if tempDis<nearDis:
            
            nearDis=tempDis
            target=row
    return target


def relief(df,dsct,ctns):
    attrs=df.columns[:-1]
    
    reliefs={}
    for attr in attrs:
        tempSigma=0.0
        for idx,row in df.iterrows():
            sameDf=df[df['code']==row['code']].copy().drop(idx,axis=0)
            nearHit=near(sameDf,dsct,ctns,row)
            diffDf=df[df['code']!=row['code']]
            nearMiss=near(diffDf,dsct,ctns,row)

            if attr in dsct:
                tempSigma=tempSigma+float(row[attr]==nearMiss[attr])**2-\
                    float(row[attr]==nearHit[attr])**2
            else:
                tempSigma=tempSigma+(row[attr]-nearMiss[attr])**2-(row[attr]-nearHit[attr])**2
        reliefs[attr]=tempSigma
    
    print(reliefs)

            


#%%
df,dsct,ctns=readWaterMelon()
relief(df,dsct,ctns)
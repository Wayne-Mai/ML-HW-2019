# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'machine learning'))
    print(os.getcwd())
except:
    pass

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import sys
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


# %%
print("hello world")


# %%
def readWaterMelon():
    df = pd.read_csv('watermelon4.csv')
    df = df.drop(df.columns[0], axis=1)
    df['code'] = 0
    return df


print(readWaterMelon())

# %%


class Kmeans:
    def __init__(self, args={}):
        self.args = args

    def train(self,df):
        for k in range(2,2+self.args['num_k']):
            self.kmeans(df,k)
        self.silhouette(df)

    def kmeans(self, df, k):
        curCenter = df.sample(n=k)
        curCenter.index = np.arange(1, k+1)
        N = df.shape[0]

        for iter_time in range(self.args['max_iter']):
            for i in range(N):
                # Find closest center vector index
                nextIdx = np.argmin(
                    np.sqrt(np.square(curCenter.iloc[:, :-1]-df.iloc[i, :-1]).sum(axis=1)))
                # reassign class label
                df.iloc[i, -1] = nextIdx

            # refresh center vector
            for j in range(k):
                curCenter.iloc[j, :-1] = df[df['code'] == j+1].mean()[:-1]

            self.args[('k', k)] = curCenter

        if k==2:
            colors=['r','g','y','b']
            ms=['+','P','x','X']
            for i in range(2):
                curDf=df[df['code']==i+1]
                curCenter=self.args[('k',2)]
                plt.scatter(curDf.iloc[:,0],curDf.iloc[:,1],marker=ms[2*i],c=colors[2*i])
                plt.scatter(curCenter.iloc[i,0],curCenter.iloc[i,1],marker=ms[2*i+1],c=colors[2*i+1])
            plt.show()

    def silhouette(self, df):
        best_k,best_avg_si=-1,-1
        N = df.shape[0]
        self.args['si']=[]
        for k in range(2, self.args['num_k']+2):
            sis=[]
            curCenter = self.args[('k', k)]

            for i in range(N):
                # in class mean distance
                ai = np.mean(np.square(
                    df[df['code'] == df['code'][i]].iloc[:, :-1]-df.iloc[i, :-1]).sum(axis=1))
                # between class min distance
                bi = float('inf')
                for code in range(1, k+1):
                    if code != df['code'][i]:
                        bi = min(bi, np.mean(
                            np.square(df[df['code'] == code].iloc[:, :-1]-df.iloc[i, :-1]).sum(axis=1)))
                # silhouette coefficient
                si=(bi-ai)/max(ai,bi)
                sis.append(si)

            # refresh best k value if cur k is better
            avg_si=sum(sis)/float(N)
            if avg_si>best_avg_si:
                best_k,best_avg_si=k,avg_si
            self.args['si'].append(avg_si)
        self.args['best_k']=best_k

# %%
df = readWaterMelon()
args={'num_k': 6, 'epsilon': 0.01, 'max_iter': 20}
model = Kmeans(args)
model.train(df)

# %%
plt.plot(np.arange(2,2+args['num_k']),model.args['si'],linestyle='--', marker='o')
plt.xlabel('K-Value')
plt.ylabel('Silhouette score')
plt.show()

#%%
df = readWaterMelon()
print(df.sum(axis=0))
print(df.sum(axis=1))

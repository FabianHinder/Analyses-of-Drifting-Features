import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
#from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import itertools
import tqdm
from boruta_py import BorutaPy as Boruta
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
import pandas as pd
#from load_data import data_loader
from scipy.sparse import block_diag, block_array, csr_matrix
import scipy.stats as stats
import scipy.sparse as sparse
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import roc_auc_score
import sys
import json
import random

def gen_data_rand_net(n=2500,ms=[15,15,15],p=0.2,ds=[5,1,0]):
    B0 = block_diag([sparse.triu(sparse.random(m, m, density=p, data_rvs=stats.norm().rvs),k=1) for m in ms])
    
    I, offset = [], 0
    for m,d in zip(ms,ds):
        I += list(np.random.choice(range(m),size=d,replace=False)+offset)
        offset += m
    B1 = csr_matrix((list(np.random.normal(size=sum(ds))), (sum(ds)*[0], I)), shape=(1,B0.shape[0]))
    
    B = block_array([[csr_matrix((1,1)),B1],[csr_matrix((sum(ms),1)),B0]])
    
    A = block_array([[sparse.eye(B.shape[0]),sparse.eye(B.shape[0])],[csr_matrix((B.shape[0],B.shape[0])),B]]).toarray()
    for _ in range(2*sum(ms)):
        A_ = A @ A
        if np.abs(A_-A).sum() <= 1e-32:
            break
        A = A_
    A = A[:sum(ms)+1,sum(ms)+2:]
    
    dist = dijkstra(B != 0, indices=[0], directed=False)[0][1:]
    #print(np.where(dist==1)[0],np.where(B1.toarray()[0,:]!=0)[0],I)
    children = set(np.where(dijkstra(B != 0, indices=[0])[0] == 2)[0]-1)
    parents = set(filter(lambda x: x not in children, np.where(dist == 2)[0]))

    T = np.random.random(size=n)
    T.sort()
    v = np.hstack( (5*np.sign(T-0.5)[:,None],np.random.normal(size=(n,sum(ms)))) )
    X = v @ A
    
    return T,X, B0,B1,A, dist,children,parents

def score_features(X,T, set_size=500, degree=5):
    T = 2*np.pi*(T-T.min())/(T.max()-T.min())
    F = np.array([np.sin(k*T-offset) for k in range(1,degree+1) for offset in [0,np.pi/2]]).T
    
    result = []
    for test_idx,train_idx in (KFold( n_splits=int(X.shape[0]/set_size) ).split(X)):
        model = ExtraTreesRegressor(max_depth=5,n_jobs=-1).fit(X[train_idx],F[train_idx])
        result.append( {
            "MI": mutual_info_regression(X[train_idx],T[train_idx]), 
            "FI": model.feature_importances_, 
            "PFI": permutation_importance(model, X[test_idx], F[test_idx], n_repeats=10, n_jobs=-1).importances_mean,
            "B": Boruta(model,early_stopping=True).fit(X,F).stat
        })
        #print(result)
    return result

if len(sys.argv) != 3:
    print("--setup n | --run_exp i")
    exit(1)
if sys.argv[1] == "--setup":
    n = int(sys.argv[2])
    setups = 100*[{"ds":[d,0,0], "p":p, "ms":[25,5,5]} for d in [1,2,3,5] for p in [0.05,0.1,0.2]]
    random.shuffle(setups)
    #print( {i: setups[i::n] for i in range(n)} )
    with open("actual_setups.json","w") as f:
        json.dump({i: setups[i::n] for i in range(n)},f)
elif sys.argv[1] == "--run_exp":
    exp_id = int(sys.argv[2])
    with open("actual_setups.json","r") as f:
        setups = json.load(f)[str(exp_id)]

    result = []
    for i,setup in enumerate(tqdm.tqdm(setups)): 
        T,X, B0,B1,A, dist,ch,pa = gen_data_rand_net(**setup)
        for j,res in enumerate(score_features(X,T)):
            for name,value in res.items():
                result.append({
                    "setup":setup,"setup_id":i,"split":j,
                    "B0":B0,"B1":B1,"A":A, 
                    "dist":dist,"ch":ch,"pa":pa,
                    "method":name,"result":value
                })
    
    pd.DataFrame(result).to_pickle("actual/result_%i.pkl.xz"%exp_id)
else:
    print("undefined!")
    exit(1)   

import numpy as np
from fcit import fcit #pip install fcit
from causaldag.utils.ci_tests.kci import ki_test_vector #pip3 install causaldag

def RIT(Y,X,p0=0.05, n_itr=5,n_samp=250):
    return RIT0(Y,X,p0,[],n_itr,n_samp)

def RIT0(Y,X,p0,S,n_itr,n_samp):
    Snew = []
    for i in range(X.shape[1]):
        if i not in S:
            if np.array([ki_test_vector(Y[sel],X[sel,i].reshape(-1,1))["p_value"] for sel in np.random.choice(range(Y.shape[0]),(n_itr,n_samp))]).mean() < p0:
                Snew.append(i)
    S.extend(Snew)
    for i in Snew:
        S.extend(RIT0(X[:,i].reshape(-1,1), X, p0, S, n_itr,n_samp))
    S = list(set(S))
    return S

def statistic_DFA(X,T=None, p0 = 5e-2, p1 = 0.1, n_perm = 15, n_itr=5,n_samp=250, verbos=False):
    if T is None:
        T = np.linspace(0,1,X.shape[0])
    if len(T.shape) == 1:
        T = T.reshape(-1,1)
    nd,fd,di = list(range(X.shape[1])),[],[]
    
    if verbos:
        print("Searching all drifting features")
    rel = RIT(T,X,p0, n_itr,n_samp)
    
    if verbos:
        print("Searching drift inducing features")
    for i_num,i in enumerate(rel):
        if verbos:
            print(i_num,"/",len(rel))
        nd.remove(i)
        R_i = np.array(np.hstack((range(0,i),range(i+1,X.shape[1]))),dtype=int)
        if fcit.test(T,X[:,i].reshape(-1,1),X[:,R_i],n_perm=n_perm) < p1:
            di.append(i)
        else:
            fd.append(i)
    return {"non-drifting":nd,"faithfully-drifting":fd,"drift-inducing":di}


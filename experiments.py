import numpy as np
import pickle

from squamish.main import Main
from statistic_DFA import statistic_DFA

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from timeit import default_timer as timer


########################### LOAD DATA ##########################################################################

rw_d1 = np.delete(np.loadtxt('data/realWorld/Elec2/elec2_data.dat'), [0,1,2,3], axis=1)
rw_d2 = np.loadtxt('data/realWorld/poker/poker.data')

t_d1 = np.loadtxt('data/theoretical/drifting-feature-analysis-1(T:1,C:5,F:15,I:5).cvs')
t_d2 = np.loadtxt('data/theoretical/drifting-feature-analysis-2(T:1,C:5,F:13,I:7).cvs')
t_d3 = np.loadtxt('data/theoretical/drifting-feature-analysis-3(T:1,C:8,F:11,I:6).cvs')
t_d4 = np.loadtxt('data/theoretical/drifting-feature-analysis-4(T:1,C:5,F:14,I:6).cvs')
t_d5 = np.loadtxt('data/theoretical/drifting-feature-analysis-5(T:1,C:11,F:8,I:6).cvs')
t_d6 = np.loadtxt('data/theoretical/drifting-feature-analysis-6(T:1,C:12,F:7,I:6).cvs')

t_l1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
t_l2 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
t_l3 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
t_l4 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
t_l5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
t_l6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

theo_data = [t_d1, t_d2, t_d3, t_d4, t_d5, t_d6]
theo_labels = [t_l1, t_l2, t_l3, t_l4, t_l5, t_l6]

'''
############################ TOY DATA EXPERIMENTS ################################################################

squamish_returns = []
dfa_returns = []
squamish_f1_scores = []
dfa_f1_scores = []
squamish_times = []
dfa_times = []

for i in range(len(theo_data)):

    data = theo_data[i]
    labels = theo_labels[i]
    X = data[:, 1:]
    y = np.linspace(0,1,X.shape[0])

    squamish_results = []
    squamish_time = []
    squamish_scores = []
    dfa_results = []
    dfa_time = []
    dfa_scores = []

    for j in range(10):

        print("THIS IS ITERATION No.: " + str(j + 1) + " ON TOY_DATA_SET No.: " + str(i + 1))

        squamish_start = timer()

        model_squamish = Main(problem_type="regression", debug=False)
        model_squamish.fit(X, y)

        squamish_end = timer()

        squamish_time.append(squamish_end - squamish_start)
        squamish_scores.append(f1_score(labels, model_squamish.relevance_classes_, average='micro'))

        dict = {}
        dict['non-drifting'] = np.where(model_squamish.relevance_classes_ == 0)[0].tolist()
        dict['faithfully-drifting'] = np.where(model_squamish.relevance_classes_ == 1)[0].tolist()
        dict['drift-inducing'] = np.where(model_squamish.relevance_classes_ == 2)[0].tolist()
        squamish_results.append(dict)

        #################################################################################################

        dfa_start = timer()

        dict = statistic_DFA(X, y)

        dfa_end = timer()

        dfa_time.append(dfa_end - dfa_start)
        dfa_results.append(dict)

        dfa_relevance_classes_ = np.zeros(len(labels))
        dfa_relevance_classes_[np.asarray(dict['faithfully-drifting'])] = 1
        dfa_relevance_classes_[np.asarray(dict['drift-inducing'])] = 2
        dfa_scores.append(f1_score(labels, dfa_relevance_classes_.tolist(), average='micro'))

    squamish_returns.append(squamish_results)
    dfa_returns.append(dfa_results)
    squamish_f1_scores.append(np.average(squamish_scores))
    dfa_f1_scores.append(np.average(dfa_scores))
    squamish_times.append(np.sum(squamish_time))
    dfa_times.append(np.sum(dfa_time))

save_data = {}

save_data['results'] = {}
save_data['time'] = {}
save_data['groundTruth'] = {}

save_data['results']['squamish'] = squamish_returns
save_data['results']['dfa'] = dfa_returns
save_data['time']['squamish'] = squamish_times
save_data['time']['dfa'] = dfa_times
save_data['groundTruth'] = theo_labels

f = open("theo_results","wb")
pickle.dump(save_data, f)
f.close()


##################################### ELECTRICITY DATA SET ######################################################

'''
X = rw_d1
y = np.linspace(0,1,X.shape[0])

scaler = StandardScaler()
X = scaler.fit_transform(X)

new1 = np.random.normal(size=(X.shape[0], ), scale=0.5 / X.std())
new2 = np.random.normal(size=(X.shape[0], ), scale=0.1 / X.std())

X = np.concatenate([X,new1.reshape(-1,1),new2.reshape(-1,1)], axis=1)

model_squamish = Main(problem_type="regression")
model_squamish.fit(X, y)

dict = {}
dict['non-drifting'] = np.where(model_squamish.relevance_classes_ == 0)[0].tolist()
dict['faithfully-drifting'] = np.where(model_squamish.relevance_classes_ == 1)[0].tolist()
dict['drift-inducing'] = np.where(model_squamish.relevance_classes_ == 2)[0].tolist()

results = statistic_DFA(X, y)

print(dict)
print(results)


################################### POKER DATA SET ##############################################################


X = rw_d2
y = np.linspace(0,1,X.shape[0])

scaler = StandardScaler()
X = scaler.fit_transform(X)

model_squamish = Main(problem_type="regression")
model_squamish.fit(X, y)

dict = {}
dict['non-drifting'] = np.where(model_squamish.relevance_classes_ == 0)[0].tolist()
dict['faithfully-drifting'] = np.where(model_squamish.relevance_classes_ == 1)[0].tolist()
dict['drift-inducing'] = np.where(model_squamish.relevance_classes_ == 2)[0].tolist()

results = statistic_DFA(X, y)

print(dict)
print(results)


import numpy as np

class random_gaussian_neuron:
    def __init__(self,n_params,n_neurons=8):
        self._n_params = n_params
        n_neurons = max(n_neurons,n_params)
        self._n_neurons = n_neurons
        self.mu_w = 2*(np.random.random(size=(n_neurons,n_params))-0.5)
        for i in range(n_params):
            self.mu_w[i,i] += 5*(np.random.randint(0,2)*2-1)
        self.mu_w = 5*np.sign(self.mu_w)*np.exp(self.mu_w**2)/( np.exp(self.mu_w**2).sum(axis=1)[:,None] )
        self.mu_b = np.random.normal(size=(n_neurons))*0.005
        self.mu_out_w = 2*(np.random.random(size=(n_neurons))-0.5)*2
        
        self.sigma_w = self.mu_w
        self.sigma_b = self.mu_b
        self.sigma_out_w = 2*(np.random.random(size=(n_neurons))-0.5)*0.1
    
    def compute(self,i):
        if len(i.shape) == 1:
            i = i.reshape(-1,1)
        assert i.shape[1] == self._n_params
        
        mu = np.inner(np.array([ np.tanh(np.inner(i,self.mu_w[k]) + self.mu_b[k]) for k in range(self._n_neurons)]).T, self.mu_out_w)
        sigma = np.inner(np.array([ np.tanh(np.inner(i,self.sigma_w[k]) + self.sigma_b[k]) for k in range(self._n_neurons)]).T, self.sigma_out_w)
        
        res = np.random.normal( size=i.shape[0] )*sigma+mu
        return (res-res.mean())/(res.std()+1e-5)

class random_drift_network:
    def __init__(self,n_inits = 12,n_neurons = 25,n_params = 6):
        self._n_inits = n_inits
        self._n_neurons = n_neurons
        
        self.paths = np.array(np.eye( n_neurons+1 ),dtype=bool)
        
        self.I = -np.ones( (n_neurons,n_params), dtype=int )
        self.n = [random_gaussian_neuron(n_params) for _ in range(n_neurons)]
        for i in range(n_neurons):
            self.I[i] = np.random.randint(0,n_inits+i,size=n_params)
            for j in self.I[i]:
                if j == 0:
                    self.paths[i+1,j],self.paths[j,i+1] =1,1
                elif j >= n_inits:
                    j = j-n_inits+1
                    self.paths[i+1,j],self.paths[j,i+1] =1,1
        self.adj = np.array(self.paths, dtype=bool)
        
        #Compute drifting features using Floyd
        for k in range(self.paths.shape[0]):
            for i in range(self.paths.shape[0]):
                for j in range(self.paths.shape[0]):
                    self.paths[i,j] = self.paths[i,j] or self.paths[i,k] and self.paths[k,j]
        self.drifting = [ i+n_inits-1 for i in range(1,self.paths.shape[0]) if self.paths[0,i]]
        
        #Find children of T (no parents exists)
        self.drift_inducing = [ i+n_inits for i in range(self.I.shape[0]) if (self.I[i] == 0).sum() > 0]
        #Add parents to obtain Markov boundary
        for i in list(self.drift_inducing):
            for j in self.I[i-n_inits]:
                if j >= n_inits:
                    self.drift_inducing.append(j)
        self.drift_inducing = list(set(self.drift_inducing))

    def compute(self,t0=-1,t1=1,n=10000):
        buf = np.empty( (self._n_inits+self._n_neurons,n) )
        buf[0,:] = np.linspace(t0,t1,n)
        buf[1:self._n_inits,:] = (np.random.random(size=(self._n_inits-1))*2-1)[:,None]
        for i in range(self._n_neurons):
            buf[i+self._n_inits] = self.n[i].compute(buf[self.I[i]].T)
        return buf.T#buf[self._n_inits:].T
    
    def get_degree(self):
        return (self.adj-np.eye(self.paths.shape[0])).sum(axis=0)[1:]
    
    def get_drift(self):
        non_drifting = [i for i in list(range(self._n_inits,self._n_inits+self._n_neurons)) if i not in self.drifting] 
        faithfully_drifting = [i for i in self.drifting if i not in self.drift_inducing]
        return {"T":[0],"params":list(range(1,self._n_inits)),"non-drifting":non_drifting,"faithfully-drifting":faithfully_drifting,"drift-inducing":list(self.drift_inducing)}

if __name__ == "__main__":
    while True:
        rdn = random_drift_network(n_inits=50)
        if len(rdn.get_drift()["drift-inducing"]) >= 5 and len(rdn.get_drift()["faithfully-drifting"]) >= 5 and len(rdn.get_drift()["non-drifting"]) >= 5 and rdn.get_degree().min() >= 1:
            break
    sample = rdn.compute()
    d = rdn.get_drift()
    I = list(d["T"]); I.extend(d["non-drifting"]); I.extend(d["faithfully-drifting"]); I.extend(d["drift-inducing"])
    print("Created new ground truth data set'./data/theoretical/drifting-feature-analysis-(T:{},C:{},F:{},I:{}).cvs'".format(len(d["T"]),len(d["non-drifting"]),len(d["faithfully-drifting"]),len(d["drift-inducing"])))
    np.savetxt("./data/theoretical/drifting-feature-analysis-(T:{},C:{},F:{},I:{}).cvs".format(len(d["T"]),len(d["non-drifting"]),len(d["faithfully-drifting"]),len(d["drift-inducing"])),sample[:,I])

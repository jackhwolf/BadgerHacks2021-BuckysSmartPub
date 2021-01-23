import numpy as np
from itertools import combinations, permutations

'''
Wrapper class around beers dataset
'''
class Data:

    def __init__(self, N=50):
        self.data = np.load('Files/beer-data/beers_processed.npy')
        self.names = np.load('Files/beer-data/beers_processed_names.npy', allow_pickle=True)
        idx = np.arange(self.data.shape[0])
        idx = np.random.choice(idx, N, replace=False)
        self.data, self.names = self.data[idx,:], self.names[idx]
        self.N, self.D = self.data.shape
        self.index = np.arange(self.N)

    def iterate_pairwise_combinations(self, idx=None):
        if idx is None:
            idx = np.arange(self.data.shape[0])
        for i, j in combinations(idx, 2):
            yield i, j, self.data[i,:], self.data[j,:]


'''
Defines logic for `customers` of data, such as clients of Buckys Smart Pub, who
provide pairwise rankings of data based on their underlying preference
'''
class DataCustomer:

    def __init__(self, data_ptr, L_dim=3, initial_R=.25):
        self.data_ptr = data_ptr
        self.L_dim = L_dim
        self.X_star = np.random.normal(size=(1,self.data_ptr.D))
        self.L_star = np.random.normal(size=(self.data_ptr.D, self.L_dim))
        self.observed = np.array([False] * self.data_ptr.N)
        sel = np.random.choice(np.arange(self.data_ptr.N), int(self.data_ptr.N*initial_R), replace=False)
        self.observed[sel] = True
        self.true_ranks = self.build_ranks()

    def pairwise_ranking_iterator(self, idx=None):
        if idx is None:
            idx = np.arange(self.data_ptr.N)
        for i, j, pi, pj in self.data_ptr.iterate_pairwise_combinations(idx):
            yield pi, pj, self.true_ranks[[i], [j]]

    def observed_ranking_iterator(self, i_idx, j_idx):
        for i in i_idx:
            for j in j_idx:
                if i == j:
                    continue
                pi, pj = self.data_ptr.data[i,:], self.data_ptr.data[j,:]
                yield pi, pj, self.true_ranks[[i], [j]]

    def scoring_iterator(self, idx=None):
        if idx is None:
            idx = self.unobserved_indexes
        for i in idx:
            yield i, self.data_ptr.data[i, :]

    @property
    def has_unobserved(self):
        return len(self.unobserved_indexes) > 0

    @property
    def observed_indexes(self):
        return np.where(self.observed)[0]

    @property
    def unobserved_indexes(self):
        return np.where(~self.observed)[0]

    def mark_observed(self, idxs):
        self.observed[idxs] = True

    def build_ranks(self):
        ranks = np.zeros((self.data_ptr.N, self.data_ptr.N))
        t_x_star = np.matmul(self.X_star, self.L_star)
        for i, j, pi, pj in self.data_ptr.iterate_pairwise_combinations():
            t_pi, t_pj = np.matmul(pi, self.L_star), np.matmul(pj, self.L_star)
            rank = self.rank_from_distance(t_pi, t_pj, t_x_star)
            ranks[i,j] = rank
        return ranks

    def rank_from_distance(self, pi, pj, x):
        dist = np.sign(np.linalg.norm(pj - x) - np.linalg.norm(pi - x))
        rank_ij = np.sign(dist) if dist != 0 else -1
        return rank_ij
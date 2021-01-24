import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

'''
siamese network to learn users preference
consists of two layers:
    - x_hat: preference vector of dimensions (1, data.D)
    - l_hat: transform vector of dimensions (data.D, L)
'''
class RankLearner:

    def __init__(self, customer_ptr, lr=0.001, wd=1e-5, epochs=1, clip_grad=0.5):
        self.D = customer_ptr.data_ptr.D
        self.L = customer_ptr.L_dim
        self.x_hat, self.l_hat = None, None
        self.initialize_layers()
        params = [self.x_hat, self.l_hat]
        self.criterion = torch.nn.MSELoss()
        self.LR = float(lr)
        self.WD = float(wd)
        self.optimizer = torch.optim.SGD(params, lr=self.LR, weight_decay=self.WD)
        self.epochs = epochs
        self.const_inp_x = torch.FloatTensor([1])
        self.const_inp_l = torch.FloatTensor(np.identity(self.D))
        torch.nn.utils.clip_grad_norm_(params, clip_grad)

    def learn_pairwise_rank(self, point_i, point_j, true_rank_ij):
        true_rank_ij = self.to_var(true_rank_ij)
        for i in range(self.epochs):
            pred_rank_ij = self.forward(point_i, point_j)
            loss = self.criterion(pred_rank_ij, true_rank_ij).sum()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def predict_pairwise_rank(self, point_i, point_j):
        with torch.no_grad():
            dist = self.forward(point_i, point_j).sign()
        dist = dist.detach().numpy().item()
        pred = np.sign(dist) if dist != 0 else -1
        return pred

    def score_point(self, point):
        point = self.to_var(point)
        with torch.no_grad():
            score = self.forward_one(point)
        return score.detach().numpy().item()

    # perform 2 forward passes and compare outputs
    def forward(self, point_i, point_j):
        point_i, point_j = self.to_var(point_i), self.to_var(point_j)
        dist_i = self.forward_one(point_i)
        dist_j = self.forward_one(point_j)
        pred_rank_ij = (dist_j - dist_i).reshape(-1)
        return pred_rank_ij

    # compute distance from point --> learned preference and compute gradients
    def forward_one(self, point):
        self.const_inp_x.matmul(self.x_hat)
        self.const_inp_l.matmul(self.l_hat)
        point_transform = point.matmul(self.l_hat)
        return (point_transform-self.x_hat.matmul(self.l_hat)).pow(2).sum() 

    def initialize_layers(self):
        self.x_hat = self.to_var(np.random.normal(size=(1,self.D)), True)
        self.l_hat = self.to_var(np.random.normal(size=(self.D, self.L)), True)

    def to_var(self, foo, rg=False):
        return Variable(torch.FloatTensor(foo), requires_grad=rg)

'''
active learning algorithm logic to learn to select points to recommend
'''
class ActiveRankLearner:

    def __init__(self, customer_ptr, checkpoint_threshold=0.85, model_kw={}):
        self.customer_ptr = customer_ptr
        self.model = RankLearner(self.customer_ptr, **model_kw)
        self.current_round = 0
        self.count = 0
        self.checkpoint_threshold = float(checkpoint_threshold)

    def step(self):
        if self.current_round == 0:
            self.learn_observed()
        elif not self.customer_ptr.has_unobserved:
            return True, {}
        else:
            idxs, scores = self.score_unobserved()
            selection = self.make_round_selection(idxs, scores)
            self.customer_ptr.mark_observed(selection)
            self.learn_observed()
        self.current_round += 1
        return self.checkpoint()

    # learn the pairwise rankings b/t observed and all other points
    def learn_observed(self):
        self.model.initialize_layers()
        obs = self.customer_ptr.observed_indexes
        idxs = self.customer_ptr.data_ptr.index
        for pi, pj, rij in self.customer_ptr.observed_ranking_iterator(obs, idxs):
            self.count += 1
            self.model.learn_pairwise_rank(pi, pj, rij)   

    # compute scoring heuristic for all unobserved points
    def score_unobserved(self):
        idxs = np.array([])
        scores = np.array([])
        for idx, point in self.customer_ptr.scoring_iterator():
            idxs = np.append(idxs, idx)
            scores = np.append(scores, self.model.score_point(point))
        return idxs.astype(int), scores

    # select point(s) to observe based on scores from this round
    def make_round_selection(self, idxs, scores):
        sidxs = np.argsort(scores)
        return idxs[[sidxs[0]]]

    def checkpoint(self):
        true = np.array([])
        pred = np.array([])
        for pi, pj, rank_ij in self.customer_ptr.pairwise_ranking_iterator():
            true = np.append(true, rank_ij)
            pred = np.append(pred, self.model.predict_pairwise_rank(pi, pj))
        current_acc = accuracy_score(true, pred)
        should_stop = current_acc >= self.checkpoint_threshold
        rep = {}
        rep['oracle_accuracy'] = current_acc
        return should_stop, rep

    def make_recommendations(self, n_rec=5):
        idxs = np.array([])
        scores = np.array([])
        for idx, point in self.customer_ptr.scoring_iterator(self.customer_ptr.data_ptr.index):
            idxs = np.append(idxs, idx)
            scores = np.append(scores, self.model.score_point(point))
        idxs = idxs.astype(int)
        sidxs = np.argsort(scores)
        name_idxs =  idxs[sidxs[:n_rec]]
        return self.customer_ptr.data_ptr.names[name_idxs]
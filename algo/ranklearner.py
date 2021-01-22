import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

class RankLearner:

    def __init__(self, customer_ptr, lr=0.00001, wd=1e-5, epochs=100, clip_grad=1):
        self.D = customer_ptr.data_ptr.D
        self.L = customer_ptr.L_dim
        self.x_hat, self.l_hat = None, None
        self.initialize_layers()
        # params = [self.x_hat, self.l_hat]
        params = [self.x_hat]
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(params, lr=float(lr), weight_decay=float(wd))
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

    def forward(self, point_i, point_j):
        point_i, point_j = self.to_var(point_i), self.to_var(point_j)
        dist_i = self.forward_one(point_i)
        dist_j = self.forward_one(point_j)
        pred_rank_ij = (dist_j - dist_i).reshape(-1)
        return pred_rank_ij

    # def forward_one(self, point):
    #     self.const_inp_x.matmul(self.x_hat)
    #     self.const_inp_l.matmul(self.l_hat)
    #     point_transform = point.matmul(self.l_hat)
    #     return (point_transform-self.x_hat.matmul(self.l_hat)).pow(2).sum() 

    def forward_one(self, point):
        self.const_inp_x.matmul(self.x_hat)
        return (point-self.x_hat).pow(2).sum() 

    def initialize_layers(self):
        self.x_hat = self.to_var(np.random.normal(size=(1,self.D)), True)
        # self.l_hat = self.to_var(np.random.normal(size=(self.D, self.L)), True)

    def to_var(self, foo, rg=False):
        return Variable(torch.FloatTensor(foo), requires_grad=rg)

class ActiveRankLearner:

    def __init__(self, customer_ptr, checkpoint_threshold=0.85, model_kw={}):
        self.customer_ptr = customer_ptr
        self.model = RankLearner(self.customer_ptr, **model_kw)
        self.current_round = 0
        self.count = 0

    def run_iterator(self):
        self.learn_observed()
        yield self.checkpoint()
        print(self.count)
        while self.customer_ptr.has_unobserved:
            self.current_round += 1
            idxs, scores = self.score_unobserved()
            selection = self.make_round_selection(idxs, scores)
            self.customer_ptr.mark_observed(selection)
            self.learn_selection(selection)
            should_stop, output = self.checkpoint()
            yield should_stop, output
            print(self.count)
            if should_stop:
                break

    def learn_observed(self):
        obs = self.customer_ptr.observed_indexes
        idxs = np.arange(len(self.customer_ptr.data_ptr.data))
        for pi, pj, rij in self.customer_ptr.observed_ranking_iterator(obs, idxs):
            self.count += 1
            self.model.learn_pairwise_rank(pi, pj, rij)

    def score_unobserved(self):
        idxs = []
        scores = []
        for idx, point in self.customer_ptr.scoring_iterator():
            idxs.append(idx)
            scores.append(self.model.score_point(point))
        return idxs, scores

    def make_round_selection(self, idxs, scores):
        sidxs = np.argsort(scores)
        idxs = np.array(idxs)[sidxs]
        return idxs[0]

    def learn_selection(self, selection):
        uobs = self.customer_ptr.unobserved_indexes
        pi = self.customer_ptr.data_ptr.data[selection,:]
        for uoidx in uobs:
            pj = self.customer_ptr.data_ptr.data[uoidx,:]
            rij = self.customer_ptr.true_ranks[[selection], [uoidx]]
            self.count += 1
            self.model.learn_pairwise_rank(pi, pj, rij) 

    def checkpoint(self):
        true = np.array([])
        pred = np.array([])
        for pi, pj, rank_ij in self.customer_ptr.pairwise_ranking_iterator():
            true = np.append(true, rank_ij)
            pred = np.append(pred, self.model.predict_pairwise_rank(pi, pj))
        current_acc = accuracy_score(true, pred)
        return current_acc >= 0.95, {'oracle_accuracy': current_acc}
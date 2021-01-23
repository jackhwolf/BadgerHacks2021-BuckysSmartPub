from algo.data import Data, DataCustomer
from algo.ranklearner import ActiveRankLearner
from uuid import uuid4
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

'''
Wrapper for customer and active learner for use by smart pub
'''
class PubCustomer:

    def __init__(self, data_ptr, customer_kw={}, model_kw={}):
        self.finished = False
        self.cid = uuid4().hex
        self.customer = DataCustomer(data_ptr, **customer_kw)
        self.active_learner = ActiveRankLearner(self.customer, **model_kw)
        self.run_iterator = self.active_learner.run_iterator()
        self.learned_rounds = 0
        self.running_drink_count = []
        self.running_oracle_accuracy = []
        self.most_recent_output = {}

    def learn(self):
        if self.finished:
            return True, {}
        should_stop, out = False, {}
        try:
            should_stop, out = next(self.run_iterator)
        except StopIteration:
            should_stop = True
        print(should_stop, out)
        self.finished = should_stop
        self.learned_rounds += 1
        self.running_drink_count.append(self.active_learner.count)
        self.running_oracle_accuracy.append(out['oracle_accuracy'])
        self.most_recent_output = out
        return should_stop, out

    @property
    def short_report(self):
        rep = OrderedDict({})
        rep['customer_id'] = self.cid
        rep['finished'] = self.finished
        rep['learned_rounds'] = self.learned_rounds
        rep['most_recent_output'] = self.most_recent_output
        rep['running_drink_count'] = self.running_drink_count
        rep['running_oracle_accuracy'] = self.running_oracle_accuracy
        rep['current_top_recommendations'] = self.active_learner.make_recommendations()
        return rep

    @property
    def long_report(self):
        data_rep = OrderedDict({})
        data_rep['N'] = self.customer.data_ptr.N
        data_rep['D'] = self.customer.data_ptr.D
        data_rep['X_star'] = self.customer.X_star.tolist()
        data_rep['L_star'] = self.customer.L_star.tolist()
        model_rep = OrderedDict({})
        model_rep['LR'] = self.active_learner.model.LR
        model_rep['WD'] = self.active_learner.model.WD
        model_rep['epochs'] = self.active_learner.model.epochs
        foo = lambda x: x.detach().numpy().tolist()
        model_rep['X_hat'] = foo(self.active_learner.model.x_hat)
        model_rep['L_hat'] = foo(self.active_learner.model.l_hat)
        rep = self.short_report
        rep['data'] = data_rep
        rep['model'] = model_rep
        return rep
    
'''
Where Bucky does his work - a smart pub with a collection of clients
Bucky visits each client and steps through the active learning process
'''
class BuckysSmartPub:

    def __init__(self, data_kw={}):
        self.data = Data(**data_kw)
        self.cid_map = {}

    def add_customer(self, customer_kw={}, model_kw={}):
        customer = PubCustomer(self.data, customer_kw, model_kw)
        self.cid_map[customer.cid] = customer
        return customer.cid

    def learn_active_customers(self, cids=None, finish=False):
        if cids is None:
            cids = list(self.cid_map)
        else:
            if not isinstance(cids, list):
                cids = [cids]
        for cid in cids:
            self.learn_active_customer(cid, finish)

    def learn_active_customer(self, cid, finish=False):
        cust = self.cid_map[cid]
        if cust.finished:
            return
        if finish:
            while not cust.finished:
                cust.learn()
        else:
            cust.learn()

    def compile_customer_reports(self, short=False):
        if short:
            return {cid: self.cid_map[cid].short_report for cid in self.cid_map}
        else:
            return {cid: self.cid_map[cid].long_report for cid in self.cid_map}

    def visualize_customer_report(self, cid, report=None,):
        if report is None:
            report = self.compile_customer_reports()[cid]

        def percs_to_str(foo):
            return [str(x) + '%' for x in (foo*100).astype(int).tolist()]

        def make_titles(report):
            suptitle = f"Customer ID: {report['customer_id']}"
            title = "sample title"
            return suptitle, title

        suptitle, title = make_titles(report)
        fig, ax = plt.subplots(figsize=(10,10))
        fig.suptitle(suptitle)
        ax.set_title(title)
        xticks = np.arange(report['learned_rounds'])
        yticks = np.linspace(0, 1, 5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_yticks(yticks)
        ax.set_yticklabels(percs_to_str(yticks))
        ax.set_ylim(0, 1)
        ax.bar(xticks, report['running_oracle_accuracy'])
        print(report)
        # plt.show()

if __name__ == "__main__":
    import json
    pub = BuckysSmartPub()
    c = pub.add_customer()

    pub.learn_active_customers(finish=True)
    pub.visualize_customer_report(c)


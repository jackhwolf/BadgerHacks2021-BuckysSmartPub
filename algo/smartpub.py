from data import Data, DataCustomer
from ranklearner import ActiveRankLearner
from uuid import uuid4
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os
import libtmux
import io

customer_names = ['jack', 'riley', 'kelley', 'scott', 'skippy', 'elvis']

'''
Wrapper for customer and active learner for use by smart pub
'''
class PubCustomer:

    def __init__(self, data_ptr, customer_kw={}, model_kw={}):
        self.finished = False
        self.cid = np.random.choice(customer_names) # uuid4().hex
        self.customer = DataCustomer(data_ptr, **customer_kw)
        self.active_learner = ActiveRankLearner(self.customer, model_kw=model_kw)
        self.learned_rounds = 0
        self.running_drink_count = []
        self.running_oracle_accuracy = []
        self.most_recent_output = {}

    def learn(self):
        if self.finished:
            return True, {}
        should_stop, out = self.active_learner.step()
        self.finished = should_stop
        self.learned_rounds += 1
        self.running_drink_count.append(self.active_learner.count)
        self.running_oracle_accuracy.append(out['oracle_accuracy'])
        self.most_recent_output = out
        return should_stop, out

    def report(self, short=True):
        if short:
            return self.short_report
        return self.long_report

    @property
    def short_report(self):
        rep = OrderedDict({})
        rep['customer_id'] = self.cid
        rep['finished'] = bool(self.finished)
        rep['learned_rounds'] = self.learned_rounds
        rep['most_recent_output'] = self.most_recent_output
        rep['running_drink_count'] = self.running_drink_count
        rep['running_oracle_accuracy'] = self.running_oracle_accuracy
        recs = self.active_learner.make_recommendations()
        rep['current_recommendations'] = recs.tolist()
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
        self.create_dask_env()

    def create_dask_env(self):
        os.system('tmux new -s blank -d')
        server = libtmux.Server()
        session_name = 'dask-smartpub-tmux'
        session = server.new_session(session_name, kill_session=True)
        os.system('tmux kill-session -t blank')
        scheduler = session.attached_window
        scheduler.rename_window('scheduler')
        workers = session.new_window('workers', attach=False)
        scheduler.attached_pane.send_keys('source venv/bin/activate')
        scheduler.attached_pane.send_keys('dask-scheduler')
        wp = workers.attached_pane
        for i in range(3):
            wp.send_keys('source venv/bin/activate')
            wp.send_keys(f'export OMP_NUM_THREADS=4; dask-worker {"192.168.86.83:8786"}')
            wp = workers.split_window(attach=False)
        workers.select_layout('tiled')
    
    def add_customer(self, customer_kw={}, model_kw={}):
        customer = PubCustomer(self.data, customer_kw, model_kw)
        self.cid_map[customer.cid] = customer
        return customer.cid

    def learn_active_customers(self, cids=None, finish=False):
        if cids is None or len(cids) == 0:
            cids = list(self.cid_map)
        for cid in cids:
            self.learn_active_customer(cid, finish)
        return cids

    def learn_active_customer(self, cid, finish=False):
        cust = self.cid_map[cid]
        if cust.finished:
            return
        if finish:
            while not cust.finished:
                cust.learn()
        else:
            cust.learn()

    def compile_customer_reports(self, cids=None, short=False):
        if cids is None:
            cids = list(self.cid_map)
        reports = []
        for cid in cids:
            cust = self.cid_map[cid]
            reports.append(cust.report(short))
        return reports

    def visualize_customer_report(self, cid, report=None, show=False):
        if report is None:
            report = self.compile_customer_reports([cid])[0]
        def percs_to_str(foo):
            return [str(x) + '%' for x in (foo*100).astype(int).tolist()]
        def make_titles(report):
            suptitle = f"Customer ID: {report['customer_id']}"
            title = "Customer Happiness by Round"
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
        ax.set_xlabel("Round")
        ax.set_ylabel("Happiness")
        ax.bar(xticks, report['running_oracle_accuracy'])
        if show:
            plt.show()
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            return buf.read() 
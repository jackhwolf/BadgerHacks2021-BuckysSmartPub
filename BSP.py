from algo.data import Data, DataCustomer
from algo.ranklearner import ActiveRankLearner
from uuid import uuid4

class BuckysSmartPub:

    def __init__(self, data_kw={}):
        self.data = Data(**data_kw)
        self.customer_ids = []
        self.customers = {}
        self.active_learners = {}
        self.finished = {}

    def add_customer(self, customer_kw={}, model_kw={}):
        cid = uuid4().hex
        customer = DataCustomer(self.data, **customer_kw)
        active_learner = ActiveRankLearner(customer, **model_kw)
        self.customer_ids.append(cid)
        self.customers[cid] = customer
        self.active_learners[cid] = active_learner.run_iterator()
        self.finished[cid] = False

    def learn_active_customers(self):
        for cid in self.customer_ids:
            if self.finished.get(cid, False):
                continue
            self.learn_active_customer(cid)
        print("====learned active customers====\n")

    def learn_active_customer(self, cid):
        customer = self.customers[cid]
        active_learner = self.active_learners[cid]
        should_stop, out = False, {}
        try:
            should_stop, out = next(active_learner)
        except StopIteration:
            should_stop = True
        print(f"{cid} stop: {should_stop}")
        print("output: ", out)
        self.finished[cid] = should_stop

if __name__ == "__main__":
    pub = BuckysSmartPub()
    pub.add_customer()

    pub.learn_active_customers()
    pub.learn_active_customers()
    pub.learn_active_customers()
    pub.learn_active_customers()
    pub.learn_active_customers()
    pub.learn_active_customers()
    pub.learn_active_customers()
    pub.learn_active_customers()
    pub.learn_active_customers()

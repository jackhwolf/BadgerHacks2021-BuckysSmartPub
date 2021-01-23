from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import json
from smartpub import BuckysSmartPub
import time
from uuid import uuid4
from distributed import Client

class DaskClient:

    def __init__(self):
        self.cli = Client('tcp://192.168.86.83:8786')
        self.cli.upload_file('data.py')
        self.cli.upload_file('ranklearner.py')
        self.cli.upload_file('smartpub.py')
        self.futures = {}
        self.n = 0
        
    def add_future(self, fn):
        fut = self.cli.submit(fn)
        fid = uuid4().hex
        self.futures[fid] = fut
        return fid

    def check_future(self, fid):
        if fid in self.futures:
            return self.futures[fid].status
        return "error"

    def gather_future(self, fid):
        return self.futures[fid].result()

    def del_future(self, fid):
        del self.futures[fid]

    def new_fid(self):
        return uuid4().hex

pub = BuckysSmartPub()
client = DaskClient()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
@app.get('/health')
def health_check():
    return f'healthy @ {int(time.time())}'

@app.get('/report')
def report(short: bool = False):
    rep = pub.compile_customer_reports(short=short)
    return JSONResponse(json.dumps(rep))

class CustomerKW(BaseModel):
    L_dim: Optional[int] = 3
    initial_R: Optional[float] = 0.25

class ModelKW(BaseModel):
    lr: Optional[float] = 0.0001
    wd: Optional[float] = 0.000001
    epochs: Optional[int] = 50
    clip_grad: Optional[float] = 0.5

class NewCustomerKW(BaseModel):
    customer: Optional[CustomerKW] = CustomerKW()
    model: Optional[ModelKW] = ModelKW()

@app.post("/add-customer")
def new_customer(data: NewCustomerKW):
    cid = pub.add_customer(dict(data.customer), dict(data.model))
    return cid

class LearnKW(BaseModel):
    cids: Optional[List[str]] = []
    finish: Optional[bool] = False

@app.post("/submit-learn")
def submit_learn(learnkw: LearnKW):
    learnkw = dict(learnkw)
    def _submit_learn():
        cids = pub.learn_active_customers(**learnkw)
        rep = pub.compile_customer_reports(cids, False)
        return rep
    fid = client.add_future(_submit_learn)
    return fid

@app.get("/check-learn")
def check_learn(fid: str):
    out = {fid: client.check_future(fid)}
    return JSONResponse(json.dumps(out))

@app.get("/gather-learn")
def check_learn(fid: str):
    out = {}
    out['fid'] = fid
    out['status'] = client.check_future(fid)
    out['result'] = {}
    if out['status'] == 'finished':
        out['result'] = client.gather_future(fid)[0]
        client.del_future(fid)
    return JSONResponse(json.dumps(out))

def deploy():
    import uvicorn
    import os
    import libtmux
    os.system('tmux new -s blank -d')
    server = libtmux.Server()
    session_name = 'api-smartpub-tmux'
    session = server.new_session(session_name, kill_session=True)
    os.system('tmux kill-session -t blank')
    api_runner = session.attached_window
    api_runner.rename_window('api_runner')
    api_runner.attached_pane.send_keys('source venv/bin/activate')
    api_runner.attached_pane.send_keys('uvicorn api:app --port=5005')

if __name__ == '__main__':
    deploy()
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import json
from smartpub import BuckysSmartPub
import time
from distributed import Client

pub = BuckysSmartPub()
app = FastAPI()
client = Client('tcp://192.168.86.83:8786', asynchronous=True)

@app.get('/')
@app.get('/health')
def health_check():
    def foo():
        return f'healthy @ {int(time.time())}'
    future = client.submit(foo)
    future = client.gather(future)
    return future

@app.get('/report')
async def report(short: bool = False):
    rep = pub.compile_customer_reports(short=short)
    return JSONResponse(json.dumps(rep))

class CustomerKW(BaseModel):
    L_dim: Optional[int] = 3
    initial_R: Optional[float] = 0.25

class ModelKW(BaseModel):
    lr: Optional[float] = 0.0001
    wd: Optional[float] = 0.000001
    epochs: Optional[int] = 1
    clip_grad: Optional[float] = 0.5

class NewCustomerKW(BaseModel):
    customer: Optional[CustomerKW] = CustomerKW()
    model: Optional[ModelKW] = ModelKW()

@app.post("/customer")
async def new_customer(data: NewCustomerKW):
    cid = pub.add_customer(dict(data.customer), dict(data.model))
    return cid

class LearnKW(BaseModel):
    cids: Optional[List[str]] = []
    finish: Optional[bool] = False

@app.post("/learn")
async def learn(learnkw: LearnKW):
    cids = pub.learn_active_customers(**dict(learnkw))
    rep = pub.compile_customer_reports(cids, True)
    return JSONResponse(json.dumps(rep))

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

def test():
    from fastapi.testclient import TestClient
    client = TestClient(app)
    print(client.get('/').json())
    # print("===============================")
    # print(client.get('/report').json())
    # print("===============================")
    # print(client.post('/customer', data=json.dumps({'customer': {}, 'model': {}})).json())
    # print("===============================")
    # print(client.post('/learn', data=json.dumps({})).json())
    # print("===============================")
    # print(client.get('/report').json())

if __name__ == '__main__':
    deploy()
    test()
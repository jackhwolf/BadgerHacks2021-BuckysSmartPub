import requests
import json
import time

url = "http://127.0.0.1:5005"

def health_check():
    return requests.get(url).json()

def report(short=False):
    return requests.get(url + f'/report?short={short}').json()

def add_customer(data={}):
    data = json.dumps(data)
    return requests.post(url + '/add-customer', data=data).json()

def submit_learn(data={}):
    data = json.dumps(data)
    req = requests.post(url + '/submit-learn', data=data)
    return req.json()

def check_learn(fid):
    req = requests.get(url + f'/check-learn?fid={fid}')
    return req.json()

def gather_learn(fid):
    req = requests.get(url + f'/gather-learn?fid={fid}')
    return req.json()

if __name__ == '__main__':
    print(health_check())
    print(report())
    print(add_customer())
    print(report())
    fid = submit_learn()
    print(fid)
    time.sleep(2)
    print(check_learn(fid))
    print(gather_learn(fid))

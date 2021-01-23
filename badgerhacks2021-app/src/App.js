import React from 'react'
import Button from 'react-bootstrap/Button';
import Alert from 'react-bootstrap/Alert';
import Card from 'react-bootstrap/Card';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';

const url = "http://127.0.0.1:5005"

class Customer extends React.Component {

  constructor(props) {
    super(props)
    this.state = {
      cid: props.cid,
      learning: null,
      result: {},
      currently_learning: false,
      recs: null,
      started: false
    }
    this.submit_learn = this.submit_learn.bind(this);
    this.gather_learn = this.gather_learn.bind(this);
  }

  async submit_learn() {
    if (this.state.currently_learning) {
      return
    }
    var resp = await fetch(url + '/submit-learn', {
      method: 'POST',
      body: JSON.stringify({cids: [this.state.cid]})
    }).then(x => x.json())
    let state = this.state
    state['learning'] = resp
    state['currently_learning'] = true
    this.setState(state)
  }

  async gather_learn() {
    var fid = this.state.learning
    var resp = await fetch(url + '/gather-learn?fid=' + fid)
      .then(x => x.json())
      .then(x => JSON.parse(x))
    if (resp['status'] === 'finished') {
      let state = this.state
      state['result'] = resp.result
      state['currently_learning'] = false
      state['started'] = true
      this.setState(state)
    } else if (resp['status'] === 'pending'){
      alert("Not finished yet!")
    }
  }

  render() {
    let rounds = this.state.result.learned_rounds
    let recs = []
    if (rounds) {
      for (var r in this.state.result.current_recommendations) {
        recs.push(<li key={r}>{this.state.result.current_recommendations[r]}</li>)
      }
    }
    let n_drinks = this.state.result.running_drink_count
    let happiness = this.state.result.running_oracle_accuracy
    return (
      <Card style={{ width: '18rem' }}>
        <Card.Body>
          <Card.Title>{this.state.cid}</Card.Title>
          <Card.Text>{rounds ? "Round #" + rounds : ""}</Card.Text>
          <Card.Text>{n_drinks ? "# drinks so far: " + n_drinks[n_drinks.length-1] : ""}</Card.Text>
          <Card.Text>{happiness ? "Current Happiness: " + happiness[happiness.length-1].toFixed(2) : ""}</Card.Text>
          {rounds ? "Current top 5 recommendations:" : ""}
          {rounds ? <ol>{recs}</ol> : ""}
        </Card.Body>
        <Card.Body>
          <Button onClick={this.submit_learn}>{this.state.started ? "Continue learning" : "Start learning"}</Button>
          <Button onClick={this.gather_learn}>Gather current learning</Button>
        </Card.Body>
      </Card>
    );
  }
}

class App extends React.Component {

  constructor(props) {
    super(props)
    this.state = {
      customers: []
    }
    this.health_check = this.health_check.bind(this);
    this.add_customer = this.add_customer.bind(this);
  }

  async health_check() {
    var resp = await fetch(url).then(x => x.json())
    console.log(resp)
  }

  async add_customer() {
    var customers = this.state.customers
    var resp = await fetch(url + '/add-customer', {
      method: 'POST', body: JSON.stringify({})
    }).then(x => x.json())
    customers.push(<Customer cid={resp} key={resp}></Customer>)
    this.setState({customers: customers})
  }

  render() {
    return (
      <div className="App">
        <Alert variant="danger">
          <Alert.Heading>Welcome to Bucky's Smart Pub</Alert.Heading>
          <p className="mb-0">Take a seat, Bucky will see you soon!</p>
          <p className="mb-0">Current # customers: {this.state.customers.length}</p>
          <Button onClick={this.add_customer}>Add Customer</Button>
        </Alert>
        <ul>{this.state.customers}</ul>
      </div>
    );
  }
}

export default App;

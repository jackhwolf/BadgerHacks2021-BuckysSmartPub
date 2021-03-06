# Buckys Smart Pub
#### Badger Hacks 2021 submission for Jack Wolf

## Overview
- Designed custom Siamese neural network for learning-to-rank/recommend by training on pairwise rankings of points
- Designed custom active learning algorithm to optimize selection of training points for this model
- Created API for users to interact with model to control training and receive recommendations
- Built frontend site for visualizing model interactions, handling multiple clients, and running model training in the background

## Buckys Smart Pub application
- Buckys Smart Pub has some beers, each of which is defined by a collection of features, e.g.
    - `Name`: `string`, label name of given beer
    - `ABV`: `float`, alcohol by volume of given beer
    - `IBU`: `float`, international bitterness unit of given beer
- The clients of BSP are all beer enthusiasts and have a preference towards a certain type of beer
- When a new client joins the Pub, Bucky's goal is to learn their preference as quickly as possible so he can 
provide quality recommendations tailored to this client
- Bucky accomplishes this by giving the newbie an initial set of a few beers to taste against each other and 
provide the pairwise rankings, called the "initial" step
    - e.g. given beers A, B, and C, the client would tell Bucky something like "I like beer A 
      more than B and C, but I like beer C more than beer B"
- Next, Bucky starts to choose beers from his shelf that he would like you pairwise rank again, called an "exploration" step
- After each step, Bucky calculates the clients "happiness level", which 
- At each step, Bucky learns more and more about the new clients preference, and hopefully within a reasonable amount 
of time and thinking has learned it pretty well

## Using a siamese neural network to learn user preference by training on pairwise rankings of points
- Data can be represented as `NxD` matrix where each point is a `1xD` vector
- User preference is composed of two parts:
    - "Preference vector", `X*`: a vector of shape `1xD` which represents their true preference
    - "Individuality factor", `L*`: a linear transform of shape `DxL` account for individuality in users, e.g. "different taste buds"
    - How to compute preference between `point_i` and `point_j`, where `+1` means `point_i > point_j`:
        ```
        -1 if norm(matmul(L*, point_i) - matmul(L*, X*)) > norm(matmul(L*, point_i) - matmul(L*, X*)), else +1
        ```
- Learning user preference with a siamese neural network composed of two layers, `X^` and `L^`
    - Training input: 
        - `point_i`, `point_j`: `1xD` vectors of points being pairwise ranked
        - `true_rank_ij`:   `scalar in {-1, +1}`, true ranking of input points
    - Predicted ranking:
        ```
        (norm(matmul(L^, point_j) - matmul(L^, X^))) - (norm(matmul(L^, point_i) - matmul(L^, X^)))
        ```
    - Compare `predicted_rank_ij` to `true_rank_ij` using loss function
    - Use backpropogation to update weights of `X^` and `L^`

## Active learning strategy to propose points for pairwise ranking
- At any given moment, there are `R` points for which we have pairwise ranked against the `N-1` other points, and our model has a learned `X^` and `L^`
- For the remaining `NR = N - R` points we have not ranked, a score can be computed using the weights of the current model
    - scoring heuristic: `s(point) = norm(matmul(L^, point) - matmul(L^, X^))`
    - Represents distance from point to current learned preference, a.k.a. how far is this point from what we think the user likes?
- checkpointing and logging results at each round:
    - after each round, we compute testing accuracy on a hold out set for pairwise ranking predictions
- Instead of training our model on all pairwise ranks in our dataset, we now 
    - Select an initial random group of `R` points and learn the pairwise ranks between `Ri, Rj for i,j in 2-combination(R)`
    - While there exists points in `NR`:
        - compute scores for all points in `NR`
        - choose point `S` which minimizes the scores
        - learn the pairwise ranks between `S, NR_P for NR_P in NR`
- Why use active learning? Minimize amount of data needed to perform well, e.g. minimize the amount of beers a client drinks before Bucky knows their preference
    - If there are `50` beers on the shelf, then there are `50!/(50-2)! = 2450` pairwise combinations of beers to try
    - Bucky could just give you all `2450` combinations to try and learn your preference with no tricks
    - Instead, Bucky wants to be smart and show you smaller selections of beers at a time
    - If Bucky starts with a selection of `5` beers and then you continue to do `3` rounds of exploration, you have roughly tried a total of `(5 * 50) + (3 * 50) = O(400)` beers
    - This can often be enough to learn accurately 

## Exposing over the web
- To visit Bucky's Smart Pub virtually, I have built a backend `FastAPI` app and frontend `React` site
- Frontend site allows people to spin up their own Smart Pub, add new clients, interact with and view the learning process, and get personalized beer recommendations
- Long-running api calls, such as calls to train the model, are ran in the background using `dask.Distributed`
- Frontend handles submitting and collecting background jobs so the page does not freeze or timeout

## To run
- clone [this repo](https://github.com/jackhwolf/BadgerHacks2021-BuckysSmartPub) which has code for frontend and implementation of algo
- in one terminal, `cd` into the `algo/` directory and run `$python3 api.py`
- in another terminal, `cd` into the `badgerhacks2021-app/` directory and run `yarn start`
- navigate to http://localhost:3000/ to join Bucky's Smart Pub virtually

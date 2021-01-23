# Buckys Smart Pub
#### Badger Hacks 2021 submission for Jack Wolf

## Overview
Active learning method for learning to recommend 

## How it works
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
- At each step, Bucky learns more and more about the new clients preference, and hopefully within a reasonable amount 
of time and thinking has learned it pretty well

## Using a siamese neural network to learn user preference by training on pairwise rankings of points
- Data can be represented as `NxD` matrix where each point is a `1xD` vector
- User preference is composed of two parts:
    - "Preference vector", `X*`: a vector of shape `1xD` which 
    - "Individuality factor", `L*`: a linear transform of shape `DxL` account for individuality in users, e.g. "different taste buds"
    - How to compute preference between `point_i` and `point_j`, where `+1 prefers point_i` and `-1 prefers point_j`:
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
    - While there exist not ranked points, `NR`:
        - compute scores for all points in `NR`
        - create exploratory set, `E`, containing the points that are argmin, argmax, and argmedian of `scores`
        - learn the pairwise ranks between `Ei, Ej for i,j in 2-combination(E)`
        - checkpoint and log
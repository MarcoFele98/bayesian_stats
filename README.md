# bayesian_stats
This comes from the workshop I attended in 2023 "Computational Summer school Modeling social and collective behaviour" (https://cosmos-konstanz.github.io/). To better learn the topics, I tried to apply some of what we learned on my own. Hence, I produced the code (and bugs). 

## Background
A group of agents is faced with the multi-armed bandit task, a classic set up for investigating the exploration-exploitation trade-off. Agents' policies (the function linking agents belives with their actions) includes both private and social information ("decision biasing"). Agents then update their belives based on the outcome of their action using reinforcement learning.

## Code

In simulate_social_RL_model.R I simulate the decison-making process. I then track the individual choices (which is the data that could be collected in a behavioural experiment) and the latent q-values, i.e., the belives each individual has regarding the quality of the available options. Such q-values would generally not be recorded as data, but it is nice to see that they tend to converge to the true option value (first plot). In the second part of the R code I interface with the stan code social_learning_model.stan, where I recover the parameters of the simulation by fitting the model with a baesian approach (MCMC) to the simulated data. The plot shows the posterior distribution for the parameters. In the last section I fit a model which does not include a social information (social_learning_model_no_gamma.stan). This is could be done to compare the two model likelihoods to perform model selection.

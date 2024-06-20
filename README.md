# bayesian_stats
This comes from the workshop I attended in 2023 "Computational Summer school Modeling social and collective behaviour" (https://cosmos-konstanz.github.io/). To better learn the topics, I tried to apply some of what we learned on my own. Hence, I produced the code (and bugs). 

## Background
A group of agents is faced with the multi-armed bandit task, a classic set up for investigating the exploration-exploitation trade-off. Agents' policies (the function linking agents belives with their actions) includes both private and social information ("decision biasing"). Agents then update their belives based on the outcome of their action using reinforcement learning.

## Code

In simulate_social_RL_model.R I simulate the decison-making process. I then track the individual choices (which is the data that could be collected in a behavioural experiment) and the latent q-values, i.e., the belives each individual has regarding the quality of the available options. Such q-values would generally not be recorded as data, but it is nice to see that they tend to converge to the true option value through reinforcment learning. 
![learning_process](https://github.com/MarcoFele98/bayesian_stats/assets/122376407/1b31f3bd-7cf4-4363-b479-a7b35aaa88df)

In the second part of the R code I interface with the stan code social_learning_model.stan, where I recover the parameters of the simulation by fitting the model with a baesian approach (MCMC) to the simulated data. The plot shows the posterior distribution for the parameters. In the last section I fit a model which does not include a social information (social_learning_model_no_gamma.stan). This could be done to compare the two model likelihoods to perform model selection.
![posterior](https://github.com/MarcoFele98/bayesian_stats/assets/122376407/5fcc30b8-0ae8-4cc9-bf8c-deb05c9ef9d9)
![posterior_2](https://github.com/MarcoFele98/bayesian_stats/assets/122376407/75ce5788-30de-4c27-bea8-64c2573bb934)


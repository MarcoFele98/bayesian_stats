# bayesian_stats
This comes from the workshop I attended in 2023 "Computational Summer school Modeling social and collective behaviour" (https://cosmos-konstanz.github.io/). I learned about reinforcement learning, social and asocial information, and the power of Bayesian data analysis to fit complex models to data. To better learn the topics, I tried to apply some of what we learned on my own. Hence, the attached code is mine (with all its problems). 

In the code I first simulate a a group of agents faced with the multi-armed bandit task. The action that they chose ("policy") is not only based on thier private information, but also on what the rest of the group is doing (which is called "decision biasing"). I then plot the dynamcis of thier individual belives. Using STAN, I then try to retrive the parameters of the decision-making model by fitting the data I have simulated. I also compare a model which does not include social information, and the likelihood of the model is lower. Nevertheless, proper model comparsion should be done using the true likelihood for every observation (not the one correct up to normalizing constant default in STAN) and should include model complexity as number of parameters.

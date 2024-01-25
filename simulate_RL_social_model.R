#_________________________________________________________________________________________________#
#___ Fitting a reinforcement learning model with social info with a Bayesian approach ____________#
#___ https://cosmos-konstanz.github.io/about/ ____________________________________________________#
#___ 10/2023 - Code (and bs) Marco Fele __________________________________________________________#
#_________________________________________________________________________________________________#
  
rm(list = ls())

library(tidyverse)
library(cowplot)
library(rstan)
library(GGally)

theme_set(theme_cowplot())

load("bayesan_stats/social learning/data.RData")

# Functions ----
softmax <- function(Qvalues, beta){
  p <- exp(beta * Qvalues)
  return(p / sum(p)) # normalize 
}

get_reward <- function(action, options){
  reward <- rnorm(1, 
        mean = options[action, "mean"], 
        sd = options[action, "sd"])
  
  return(reward)
}

# Model _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________----
# I could be generating this data with STAN (much faster) with expose_stan_functions() https://notebook.community/QuantEcon/QuantEcon.notebooks/IntroToStan_basics_workflow
simulate_decision_biasing <- function(n_agents, 
                                      options,
                                      max_time, 
                                      alpha, beta, # individual parameters
                                      gamma, theta # social learning parameters
) {
  #browser()
  n_actions <- nrow(options)
  
  # data containers 
  current_Qvalues <- matrix(0, 
                            ncol = n_actions, 
                            nrow = n_agents)
  save_data <- matrix(0, 
                      ncol = n_actions + 4, # columns are Qvalues, time, agent number, action, reward
                      nrow = n_agents * max_time) 
  
  for(time in 1:max_time) {
    social_choices <- rep(0, n_actions) # Keeps tracks of social choices 
    agents <- sample(1:n_agents, n_agents, replace = F) # randomly choose the order
    for(agent_counter in 1:n_agents) {
      agent <- agents[agent_counter] # chose acting agent
      # Calculate probability of individual choice 
      individual_choice <- softmax(Qvalues = current_Qvalues[agent, ], beta = beta)
      # Calculate probability of group choice 
      group_choice <- individual_choice
      if(agent_counter != 1) {
        social_frequencies <- (social_choices/(agent_counter-1))^theta 
        group_choice <- social_choices/sum(social_frequencies) 
      }
      # Chosen action
      probabilities <- (1 - gamma) * individual_choice + gamma * group_choice
      if(sum(probabilities) == 0) {
        probabilities <- rep(1, n_actions)
      }
      action <- sample(1:n_actions, 1, prob = probabilities)
      # get reward
      reward <- get_reward(action, options)
      # update Q values
      current_Qvalue <- current_Qvalues[agent, action]
      current_Qvalues[agent, action] <- current_Qvalue + alpha * (reward - current_Qvalue)
      # update social choices counter
      social_choices[action] <- social_choices[action] + 1
      # save results
      save_data[(time - 1) * n_agents + agent_counter, ] <- c(current_Qvalues[agent, ], time, agent, action, reward)
    }
  }
  colnames(save_data) <- c(1:n_actions, "time", "agent", "chosen_action", "reward")
  return(save_data)
}

# Simulate data _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________----

## Parameters ----
# Global parameters
n_agents <- 10
max_time <- 150
n_replicates <- 10

# Decision-biasing reinforcement learning parameters 
alpha <- 0.5
beta <- 0.6
gamma <- 0.9 
theta <- 1

parameters <- data.frame(parameter = c("alpha", "beta", "gamma", "theta"),
                                value = c(alpha, beta, gamma, theta))

# Decision task parameters
n_options <- 3
options <- matrix(c(seq(1, 10, l = n_options), # options means
                    rep(5, n_options), # options sd
                    1:n_options), 
                  byrow = F,
                  ncol = 3)
colnames(options) <- c("mean", "sd", "action")


## Simulate mixed learning ----
data_mixed_learning <- data.frame() # columns are Qvalues, time, agent number, action, reward, replicate)
for(replicate in 1:n_replicates) {
  print(replicate)
  data_mixed_learning <- rbind(data_mixed_learning,
                               as.data.frame(cbind(simulate_decision_biasing(n_agents = n_agents, 
                                                                             options = options,
                                                                             max_time = max_time, 
                                                                             alpha = alpha, # individual parameters
                                                                             beta = beta, 
                                                                             gamma = gamma, # social parameters
                                                                             theta = theta 
                               ), 
                               replicate)))
}

data_mixed_learning_l <- pivot_longer(data_mixed_learning, 
                                      cols = 1:nrow(options),
                                      names_to = "action",
                                      values_to = "qValue") |>
  group_by(replicate, time) |>
  mutate(avg_rewards = mean(reward))|>
  group_by(replicate, action, time) |>
  mutate(avg_qValue = median(qValue))


## Visualize data ----
# Q-values
ggplot(data_mixed_learning_l) +
  geom_line(aes(time, qValue, 
                color = as.factor(action),
                group = interaction(agent, action)),
            alpha = 0.3) +
  geom_line(aes(time, avg_qValue,
                color = as.factor(action)),
            linewidth = 2) +
  geom_hline(data = as.data.frame(options),
             aes(yintercept = mean,
                 color = as.factor(action)), 
             lty = 2, linewidth = 1) +
  geom_smooth(aes(time, reward), lty = 2,
              linewidth = 2, color = "black") +
  facet_wrap(~replicate) +
  scale_color_discrete(name = "Action") +
  background_grid()

# rewards
ggplot(data_mixed_learning_l) +
  geom_line(aes(time, reward, 
                color = as.factor(agent)),
            alpha = 0.5) +
  geom_smooth(aes(time, reward), 
            linewidth = 2, color = "black") +
  facet_wrap(~replicate) +
  background_grid()


# STAN ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________----

# Compile 
model <- stan_model(file = "bayesan_stats/social learning/social_learning_model.stan", # compile to create .rds file
                    model_name = "bayesan_stats/social learning/social_learning_model",
                    auto_write = T) 

# Alternative
# fit_social <- stan(file = "social_learning_model.stan", 
#                    data = data_social, 
#                    #cores = 4, 
#                    chains = 4, 
#                    iter = 2000)

# Create data
data_mixed <- list(
  n_replicates = n_replicates,
  n_trials = max_time,
  n_agents = n_agents,
  n_actions = nrow(options),
  agent_sequence = lapply(split(data_mixed_learning, 
                                data_mixed_learning$replicate), function(x) { 
                                  mutate(x, agent_order = rep(1:n_agents, times = max_time)) |>
                                    pivot_wider(id_cols = "time",
                                                names_from = "agent_order",
                                                values_from = "agent") |>
                                    select(-time) |>
                                    as.matrix()}) |> 
    unlist() |> 
    array(dim = c(max_time, n_agents, n_replicates)),
  choices = lapply(split(data_mixed_learning, 
                         data_mixed_learning$replicate), function(x) { 
                           arrange(x, agent) |>
                             pivot_wider(id_cols = "time",
                                         names_from = "agent",
                                         values_from = "chosen_action") |>
                             select(-time) |>
                             as.matrix()})|> 
    unlist() |> 
    array(dim = c(max_time, n_agents, n_replicates)),
  rewards = lapply(split(data_mixed_learning, 
                         data_mixed_learning$replicate), function(x) { 
                           arrange(x, agent) |>
                             pivot_wider(id_cols = "time",
                                         names_from = "agent",
                                         values_from = "reward") |>
                             select(-time) |>
                             as.matrix()})|> 
    unlist() |> 
    array(dim = c(max_time, n_agents, n_replicates)))

# MCMC sampling
fit_mixed <- sampling(model, # MCMC sampling
                      data = data_mixed, 
                      cores = 4, # takes a while
                      chains = 4, 
                      iter = 2000)

# Visualize result
draws_mixed <- as.data.frame(fit_mixed)
draws_mixed_l <- draws_mixed |>
  select(-lp__) |>
  pivot_longer(cols = 1:4,
               names_to = "parameter",
               values_to = "value")

ggplot(draws_mixed_l) +
  geom_histogram(aes(value)) +
  geom_vline(data = parameters,
             aes(xintercept = value)) +
  facet_wrap(~parameter)

ggpairs(draws_mixed, aes(alpha = 0.1),
        upper = list(continuous = "density"))


# Model comparison __________________________________________________________________________________________________________________________________________________________________________________________________________________----

# fit model
model_no_gamma <- stan_model(file = "bayesan_stats/social learning/social_learning_model_no_gamma.stan", # compile to create .rds file
                    model_name = "bayesan_stats/social learning/social_learning_model_no_gamma",
                    auto_write = T)

# all parameters
fit_comparison <- sampling(model_no_gamma, # MCMC sampling
                      data = data_mixed, 
                      cores = 4, # takes a while
                      chains = 4, 
                      iter = 2000)

# Visualize result
draws_comparison <- as.data.frame(fit_comparison)
draws_comparison_l <- draws_comparison |>
  select(-lp__) |>
  pivot_longer(cols = 1:2,
               names_to = "parameter",
               values_to = "value")

ggplot(draws_comparison_l) + # shit parameters as expected
  geom_histogram(aes(value)) +
  geom_vline(data = parameters |> filter(!(parameter %in% c("gamma", "theta"))),
             aes(xintercept = value)) +
  facet_wrap(~parameter)

ggpairs(draws_comparison, aes(alpha = 0.1),
        upper = list(continuous = "density"))
# Values of likelihood much lower for this model (without social influence) than for the other model (with social influence). Conclusion: more likely to be social influence (and this is the case since I simulated the data). 
# (To do things properly you also have to penalize for the number of parameters).
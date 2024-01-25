//_________________________________________________________________________________________________//
//___ Fitting a reinforcement learning model with social info with a Bayesian approach ____________//
//___ https://cosmos-konstanz.github.io/about/ ____________________________________________________//
//___ 10/2023 - Code (and bs) Marco Fele __________________________________________________________//
//_________________________________________________________________________________________________//

data {
  int <lower = 1> n_replicates; // <> for bounds
  int <lower = 1> n_trials; 
  int <lower = 1> n_agents;
  int <lower = 1> n_actions;

  int <lower = 1, upper = n_agents> agent_sequence[n_trials, n_agents, n_replicates]; // Column is order of choices made, entry is aget id. Matrices can hold only real, and so I have to use array for integers
  int <lower = 1, upper = n_actions> choices[n_trials, n_agents, n_replicates]; 
  //matrix <lower = 0> [n_trials, n_agents, n_replicates] rewards; 
  real  rewards[n_trials, n_agents, n_replicates]; 
}

parameters {
  real <lower = 0, upper = 1> alpha;
  real <lower = 0> beta;
  //real <lower = 0, upper = 1> gamma;
  //real <lower = 0, upper = 10> theta; // if boundaries are not specified, we get divergent transitions: energy of "marble" is not conserved during "marble roll" (transition). 
}

model {
  // Priors 
  alpha ~ beta(1, 1);
  beta ~ uniform(0, 10);
  real gamma = 0;
  //gamma ~ beta(1, 1);
  real theta = 1;
  //theta ~ uniform(0, 10);
  
  for(replicate in 1:n_replicates) {
    // Every variable has to be declared, and when a matrix is declared its dimensions have to be specified
    // matrix[n_agents, n_actions] current_Qvalues; declaration
    // current_Qvalues = rep_matrix(0, n_agents, n_actions); definition
    matrix[n_agents, n_actions] current_Qvalues = rep_matrix(0, n_agents, n_actions);

    for(trial in 1:n_trials) {
      vector[n_actions] social_choices = rep_vector(0, n_actions); // counter for what the others have done
      for(agent_order_number in 1:n_agents) {
        int agent = agent_sequence[trial, agent_order_number, replicate];
        // calculate probability of individual choice
        vector[n_actions] individual_qValues = to_vector(sub_row(current_Qvalues, agent, 1, n_actions)); // convert from row vector to vector
        vector[n_actions] individual_choice = softmax(individual_qValues * beta);
        //vector[n_actions] individual_choice = exp(individual_qValues * beta) / sum(exp(individual_qValues * beta)); // alternative option

        // calculate probability of group choice
        vector[n_actions] group_choice = individual_choice; // rep_vector(1, n_actions) / n_actions;
        if(agent_order_number != 1) { // if its not the first of the group to make a decision, group_choice is really based on the group
          vector[n_actions] social_frequencies = social_choices / (agent_order_number - 1); // exclude focus agent
          for(i in 1:n_actions) {
            social_frequencies[i] = pow(social_frequencies[i] + 0.0001, theta); // pow(x, y) is not vectorized... I have to add the small number + 0.0001 because 0^0 issues I think
          }
          group_choice = social_frequencies / sum(social_frequencies);
        }
        // make choice
        vector[n_actions] actions_probabilites = (1 - gamma) * individual_choice + gamma * group_choice;
        actions_probabilites = actions_probabilites / sum(actions_probabilites); // normalize

        // THIS IS THE MOST IMPORTANT PART. This is the likelihood P(data|parameters). Here the "generative" nature of the model is included through a probability density function (probability mass function for discrete outcomes). In this "complex" model, lk is also dependent on the state of the individual, specifically the q_values, since the action probabilities depend on the previous timesteps (and the model parameters). Its as if "likelihood = P(data|parameter,state), with state depending on data and parameters of previous time step".
        target += categorical_lpmf(choices[trial, agent, replicate] | actions_probabilites);
        //choices[trial, agent, replicate] ~ categorical(actions_probabilites); Alternative: With the ~ notation, normalizing constants are dropped, and the lp__ parameter is correct up to that constant and useless for model comparison. This notation is faster only with likelihoods with lots of normalizing constants.
        int action = choices[trial, agent, replicate];

        // update social choices counter
        social_choices[action] += 1;

        // update latent Q-values
        real reward = rewards[trial, agent, replicate];
        real reward_prediction_error = reward - current_Qvalues[agent, action];
        current_Qvalues[agent, action] += alpha * reward_prediction_error;
      }
    }
  }
}

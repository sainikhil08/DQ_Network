In the training phase, I ran the model with 20 episodes each with 20 steps.

training_log.txt file shows the reward results for each episode in training phase and also TD error for each DQN update


In testing Phase, I ran the model for 100 episodes with 10 steps in each episode.
this test agent uses trained DQ network

test_results.txt file shows results for each episode whether its well mixed or not and its reward.

instructions:

Run the exec_environment file, it will start training the model then executes
test_agent to test the model
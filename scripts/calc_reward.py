import pandas as pd
import numpy as np
mode_dir = './results/StagHuntGW/paper-505/run1/eval_finetune/res'
rewards = []
agent0_rewards = []
agent1_rewards = []
for i in range(20):
    rewards_ep_df = pd.read_csv(f"{mode_dir}/episode{i}/rewards.csv")
    rewards_ep_agent0 = rewards_ep_df['agent0'].values
    rewards_ep_agent1 = rewards_ep_df['agent1'].values
    tot_reward_agent0 = np.sum(rewards_ep_agent0)
    tot_reward_agent1 = np.sum(rewards_ep_agent1)
    agent0_rewards.append(tot_reward_agent0)
    agent1_rewards.append(tot_reward_agent1)
    rewards.append(tot_reward_agent0+tot_reward_agent1)
print("mean:", np.mean(rewards))
print(np.std(rewards))  
print("mean:", np.mean(agent0_rewards))
print(np.std(agent0_rewards))
print("mean:", np.mean(agent1_rewards))
print(np.std(agent1_rewards))
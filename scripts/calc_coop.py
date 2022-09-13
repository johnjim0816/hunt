from tensorboard.backend.event_processing import event_accumulator
import numpy as np
run_id = 3
file_name = 'events.out.tfevents.1662631842'
file_suffix = '.jiangji-policy-diversity-1-76cd66576b-xcmv4'
model_dir = f"./results/StagHuntGW/paper-new1_back/run{run_id}/eval_finetune/logs/coop&coop_num_per_episode/coop&coop_num_per_episode"
ea=event_accumulator.EventAccumulator(f"{model_dir}/{file_name}{file_suffix}") 
ea.Reload()

coop_items = ea.scalars.Items("coop_coop_num_per_episode")
# entropies = ea.scalars.Items(f"agent{agent_id}/dist_entropy")
selected_coops = [coop_items[i].value for i in range(len(coop_items))]
print(np.mean(selected_coops ))
print(np.std(selected_coops ))

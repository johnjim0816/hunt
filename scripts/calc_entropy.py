from tensorboard.backend.event_processing import event_accumulator
import numpy as np
run_id = 3
agent_id = 1
file_name = 'events.out.tfevents.1662017492.jiangji-policy-diversity-1-76cd66576b-xcmv4'
model_dir = f"./results/StagHuntGW/paper-505/run{run_id}/logs/agent{agent_id}/dist_entropy/dist_entropy"
ea=event_accumulator.EventAccumulator(f"{model_dir}/{file_name}") 
ea.Reload()

entropies = ea.scalars.Items(f"agent{agent_id}/dist_entropy")
selected_entropies = [entropies[i].value for i in range(1500,2000)]
print(np.mean(selected_entropies))
print(np.std(selected_entropies))
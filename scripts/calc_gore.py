from tensorboard.backend.event_processing import event_accumulator
import numpy as np
run_id = 1
gore_id = 1
file_name = 'events.out.tfevents.1662628099'
file_suffix = '.jiangji-policy-diversity-1-76cd66576b-xcmv4'
model_dir1 = f"./results/StagHuntGW/paper-505_back/run{run_id}/eval_finetune/logs/gore{gore_id}_num_per_episode/gore{gore_id}_num_per_episode"
ea=event_accumulator.EventAccumulator(f"{model_dir1}/{file_name}{file_suffix}") 
ea.Reload()

gore1_items = ea.scalars.Items("gore1_num_per_episode")

selected_gore1 = [gore1_items[i].value for i in range(len(gore1_items))]
print(np.mean(selected_gore1))
print(np.std(selected_gore1))
gore_id = 2
model_dir2 = f"./results/StagHuntGW/paper-505_back/run{run_id}/eval_finetune/logs/gore{gore_id}_num_per_episode/gore{gore_id}_num_per_episode"
ea=event_accumulator.EventAccumulator(f"{model_dir2}/{file_name}{file_suffix}") 
ea.Reload()

gore2_items = ea.scalars.Items("gore2_num_per_episode")

selected_gore2 = [gore2_items[i].value for i in range(len(gore2_items))]
print(np.mean(selected_gore2))
print(np.std(selected_gore2))
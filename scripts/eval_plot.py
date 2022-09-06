
from tensorboard.backend.event_processing import event_accumulator
 
#加载日志数据
ea=event_accumulator.EventAccumulator('./results/StagHuntGW/paper-505/run5/eval_finetune/logs/events.out.tfevents.1662087863.jiangji-policy-diversity-1-76cd66576b-xcmv4') 
ea.Reload()
print(ea.scalars.Keys())

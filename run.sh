#!/bin/bash

# height exp
for reg in {'[eye_loss_height(0)]','[eye_loss_height(0.5)]'}
do
    python main.py 'eye_height_exp' $reg
done

# only using expert feature
python main.py 'expert_feature_only_exp'

# random risk
for reg in {'[eye_loss]','[eye_loss2]'}
do
    python main.py 'random_risk_exp' $reg
done

# compare differnet regularizations
for reg in {'[eye_loss]','[eye_loss2]','[lasso]','[enet]','[wridge,wridge1_5,wridge3]','[wlasso,wlasso1_5,wlasso3]'}
do
    python main.py 'reg_exp' $reg
done



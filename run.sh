#!/bin/bash

# python main.py 'reg_exp' '[owl]'

# only using expert feature
python main.py 'expert_feature_only_exp'

# random risk
for reg in {'[eye_loss]'}
do
    python main.py 'random_risk_exp' $reg
done

# compare differnet regularizations
for reg in {'[eye_loss]','[ridge]','[lasso]','[enet]','[wridge,wridge1_5,wridge3]','[wlasso,wlasso1_5,wlasso3]','[owl]'}
do
    python main.py 'reg_exp' $reg
done

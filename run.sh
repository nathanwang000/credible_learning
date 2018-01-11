#!/bin/bash

python main.py 'random_risk_exp'

python main.py 'expert_feature_only_exp'

# compare differnet regularizations
for reg in {'[eye_loss]','[lasso]','[enet]','[owl]','[wridge,wridge1_5,wridge3]','[wlasso,wlasso1_5,wlasso3]'}
do
    python main.py 'reg_exp' $reg
done


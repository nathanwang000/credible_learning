#!/bin/bash

# cbm
python main.py 'expert_feature_only_exp'

# stdcx
python main.py 'stdcx_shortcut_exp'

# compare different regularizations # ridge is std(x)
for reg in {'[eye_loss]','[ridge]',}
do
    python main.py 'shortcut_exp' $reg
done

# ccm res
python main.py 'res_shortcut_exp' '["model_expert_only/testexpert_only_ridge^0.1.pt"]'


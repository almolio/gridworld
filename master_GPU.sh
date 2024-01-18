#!/bin/bash
#SBATCH --mail-user=ge95kag@mytum.de
VARIANT=0
BATCH_SIZE=(128)
EPISODE_LENGTH=(200)
EPS_DECAY=(1000000)
# EPS_START=(0.9, 0.95, 0.99)
EPS_START=(0.99)
# EPS_START=(0.3)
DROP_OUT=(0.0)
# DROP_OUT=(0.0)
EPS_END=(0.01)
# GAMMA=(0.95, 0.98, 0.995)
GAMMA=(0.98)
layer_dim=(128)
LR=(0.0001)
LR_DECAY=(1000)
MEMORY_SIZE=(15000)
n_actions=(5)
NUM_EPISODES=(5002)
num_hidden_layers=(10)
TARGET_UPDATE_PERIOD=(9000, 11000)
TAU=(1)
VAL_CHECK_PERIOD=(150)
VAL_NB_EPISODES=(20)
for BATCH_SIZE in "${BATCH_SIZE[@]}"
do for DROP_OUT in "${DROP_OUT[@]}"
do for EPISODE_LENGTH in "${EPISODE_LENGTH[@]}"
do for EPS_DECAY in "${EPS_DECAY[@]}"
do for EPS_START in "${EPS_START[@]}"
do for EPS_END in "${EPS_END[@]}"
do for GAMMA in "${GAMMA[@]}"
do for layer_dim in "${layer_dim[@]}"
do for LR in "${LR[@]}"
do for LR_DECAY in "${LR_DECAY[@]}"
do for MEMORY_SIZE in "${MEMORY_SIZE[@]}"
do for n_actions in "${n_actions[@]}"
do for NUM_EPISODES in "${NUM_EPISODES[@]}"
do for num_hidden_layers in "${num_hidden_layers[@]}"
do for TARGET_UPDATE_PERIOD in "${TARGET_UPDATE_PERIOD[@]}"
do for TAU in "${TAU[@]}"
do for VAL_CHECK_PERIOD in "${VAL_CHECK_PERIOD[@]}"
do for VAL_NB_EPISODES in "${VAL_NB_EPISODES[@]}"
do 
sbatch --job-name="test" --export=BATCH_SIZE=$BATCH_SIZE,VARIANT=$VARIANT,DROP_OUT=$DROP_OUT,GAMMA=$GAMMA,EPS_START=$EPS_START,EPS_END=$EPS_END,EPS_DECAY=$EPS_DECAY,TAU=$TAU,LR=$LR,LR_DECAY=$LR_DECAY,NUM_EPISODES=$NUM_EPISODES,MEMORY_SIZE=$MEMORY_SIZE,TARGET_UPDATE_PERIOD=$TARGET_UPDATE_PERIOD,EPISODE_LENGTH=$EPISODE_LENGTH,VAL_CHECK_PERIOD=$VAL_CHECK_PERIOD,VAL_NB_EPISODES=$VAL_NB_EPISODES,n_actions=$n_actions,num_hidden_layers=$num_hidden_layers,layer_dim=$layer_dim run_test_DRL.sbatch
printf "$JOB_ID"
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
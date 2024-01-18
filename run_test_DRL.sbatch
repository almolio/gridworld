#!/bin/bash
#SBATCH -N 1
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -o out_test_run.out
#SBATCH -e err_test_run.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --time=8:00:00
source newdrl_env/bin/activate    # to activate virtual environment
python main.py --batch_size $BATCH_SIZE --variant $VARIANT --dropout_rate $DROP_OUT --gamma $GAMMA --eps_start $EPS_START --eps_end $EPS_END --eps_decay $EPS_DECAY --tau $TAU --lr $LR --lr_decay $LR_DECAY --num_episodes $NUM_EPISODES --memory_size $MEMORY_SIZE --target_update_period $TARGET_UPDATE_PERIOD --episode_length $EPISODE_LENGTH --val_check_period $VAL_CHECK_PERIOD --val_nb_episodes $VAL_NB_EPISODES --n_actions $n_actions --num_hidden_layers $num_hidden_layers --layer_dim $layer_dim 
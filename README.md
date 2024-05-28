# The official implementation of CityLight: A Unified Model Towards Real-world City-scale Traffic Signal Control Coordination

Due to the large size of the dataset file, we have stored it in https://drive.google.com/drive/folders/1LxsdybT5Ai1AUjGT4m6Ku-951zEehU7Y?usp=drive_link.

## configs to reproduce our results:
1) Chaoyang
python run_ppo.py --data data/beijing_chaoyang --steps 3600 --lr 5e-4 --critic_lr 5e-4 --buffer_episode_size 60 --cuda_id 0 --ppo_epoch 16 --training_step 10000000 --algo mp_builtin --interval 15 --agg 1 --attn type_attn --distance 2 --reward one_hop_queue --num_mini_batch 6 --layer_N 2 
2) Central Beijing
python run_ppo.py --data data/central_beijing --steps 3600 --lr 5e-4 --critic_lr 5e-4 --buffer_episode_size 12 --cuda_id 0 --ppo_epoch 16 --training_step 10000000 --algo mp_builtin --interval 15 --agg 1 --attn type_attn --distance 2 --reward one_hop_queue --num_mini_batch 4 --layer_N 2 
3) Jinan
python run_ppo.py --data data/jinan --steps 3600 --lr 5e-4 --critic_lr 5e-4 --buffer_episode_size 12 --cuda_id 0 --ppo_epoch 10 --training_step 10000000 --algo mp_builtin --interval 15 --agg 1 --attn type_attn --distance 2 --reward one_hop_queue --num_mini_batch 18 --layer_N 2 
4) Beijing
python run_ppo.py --data data/beijing --steps 3600 --lr 5e-4 --critic_lr 5e-4 --buffer_episode_size 4 --cuda_id 0 --ppo_epoch 10 --training_step 10000000 --algo mp_builtin --interval 15 --agg 1 --attn type_attn --distance 2 --reward one_hop_queue --num_mini_batch 18 --layer_N 2 



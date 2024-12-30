export HF_TOKEN=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

cd ~/fork/llama-recipes/

export DATASET_PATH=leave_wave_34_out_remaining_9_1_0_split
export STEERING_TYPE=QA

export MODEL_NAME="/rscratch/data/llama-hf/hub/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
export MODEL_NICKNAME="Meta-Llama-2-13B"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 recipes/quickstart/finetuning/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --use_peft --peft_method='lora' --use_fp16 --mixed_precision --batch_size_training 32 --val_batch_size 32 --gradient_accumulation_steps 1 --dist_checkpoint_root_folder ~/llama-recipes/outputs/llama2/ --dist_checkpoint_folder llama2 --dataset opnqa_steering_dataset --batching_strategy='padding' --output_dir /rscratch/data/steerable_pluralism/lora/${DATASET_PATH}/${STEERING_TYPE}_${MODEL_NICKNAME} --name ${DATASET_PATH}_${STEERING_TYPE}_${MODEL_NICKNAME} --lr 4e-4 --num_epochs 10 --loss_function_type wd --dataset_path ${DATASET_PATH} --steering_type ${STEERING_TYPE} --model_name ${MODEL_NAME}

#  export MODEL_NAME="/rscratch/data/llama-hf/hub/models--meta-llama--Meta-Llama-3-70B/snapshots/c82494877ce7f6d7d317c56ec081328e382c72fe"
# CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nnodes=1 --nproc-per-node=4 recipes/quickstart/finetuning/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --use_peft --peft_method='lora' --use_fp16 --mixed_precision --batch_size_training 8 --val_batch_size 8 --gradient_accumulation_steps 4 --dist_checkpoint_root_folder ~/llama-recipes/outputs/llama2/ --dist_checkpoint_folder llama2 --dataset opnqa_steering_dataset --batching_strategy='padding' --output_dir /rscratch/data/steerable_pluralism/lora/${DATASET_PATH}/${STEERING_TYPE}_Meta-Llama-3-70B --name ${DATASET_PATH}_${STEERING_TYPE}_Meta-Llama-3-70B --lr 4e-4 --num_epochs 10 --loss_function_type wd --dataset_path ${DATASET_PATH} --steering_type ${STEERING_TYPE} --model_name ${MODEL_NAME}

# export MODEL_NAME="/rscratch/data/llama-hf/hub/models--meta-llama--Meta-Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac"
# torchrun --nnodes=1 --nproc-per-node=8 recipes/quickstart/finetuning/finetuning.py --enable_fsdp --fsdp_config.pure_bf16 --use_peft --low_cpu_fsdp --peft_method='lora' --use_fp16 --mixed_precision --batch_size_training 8 --val_batch_size 8 --gradient_accumulation_steps 4 --dist_checkpoint_root_folder ~/llama-recipes/outputs/llama2/ --dist_checkpoint_folder llama2 --dataset opnqa_steering_dataset --batching_strategy='padding' --output_dir /rscratch/data/steerable_pluralism/lora/${DATASET_PATH}/${STEERING_TYPE}_Meta-Llama-3p1-70B --name ${DATASET_PATH}_${STEERING_TYPE}_Meta-Llama-3p1-70B --lr 4e-4 --num_epochs 10 --loss_function_type wd --dataset_path ${DATASET_PATH} --steering_type ${STEERING_TYPE} --model_name ${MODEL_NAME}
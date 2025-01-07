export HF_TOKEN=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

cd ~/fork/llama-recipes/ 

export DATASET_PATH=leave_wave_34_out_remaining_9_1_0_split
export MODEL_NAME="/rscratch/data/llama-hf/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
export MODEL_NICKNAME=llama3p1_8b_pretrained

export STEERING_TYPE=QA

# lr tuning
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 \
--master_port=29507 recipes/quickstart/finetuning/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --use_peft --peft_method='lora' --use_fp16 --mixed_precision \
--batch_size_training 32 --val_batch_size 32 --gradient_accumulation_steps 1 \
--dist_checkpoint_root_folder ~/llama-recipes/outputs/llama2/ --dist_checkpoint_folder llama2 --batching_strategy='padding' \
--dataset opnqa_steering_dataset  --output_dir /rscratch/data/steerable_pluralism/lora/${DATASET_PATH}/${STEERING_TYPE}/${MODEL_NICKNAME}/ \
--name ${DATASET_PATH}_${STEERING_TYPE}_${MODEL_NICKNAME} --dataset_path ${DATASET_PATH} --steering_type ${STEERING_TYPE} --model_name ${MODEL_NAME} \
--lr 5e-5 --num_epochs 10 --loss_function_type wd --which_scheduler='step' --warmup_ratio 0.10 --gamma 0.85

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 \
--master_port=29507 recipes/quickstart/finetuning/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --use_peft --peft_method='lora' --use_fp16 --mixed_precision \
--batch_size_training 32 --val_batch_size 32 --gradient_accumulation_steps 1 \
--dist_checkpoint_root_folder ~/llama-recipes/outputs/llama2/ --dist_checkpoint_folder llama2 --batching_strategy='padding' \
--dataset opnqa_steering_dataset  --output_dir /rscratch/data/steerable_pluralism/lora/${DATASET_PATH}/${STEERING_TYPE}/${MODEL_NICKNAME}/ \
--name ${DATASET_PATH}_${STEERING_TYPE}_${MODEL_NICKNAME} --dataset_path ${DATASET_PATH} --steering_type ${STEERING_TYPE} --model_name ${MODEL_NAME} \
--lr 2e-4 --num_epochs 10 --loss_function_type wd --which_scheduler='step' --warmup_ratio 0.10 --gamma 0.85

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 \
--master_port=29507 recipes/quickstart/finetuning/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --use_peft --peft_method='lora' --use_fp16 --mixed_precision \
--batch_size_training 32 --val_batch_size 32 --gradient_accumulation_steps 1 \
--dist_checkpoint_root_folder ~/llama-recipes/outputs/llama2/ --dist_checkpoint_folder llama2 --batching_strategy='padding' \
--dataset opnqa_steering_dataset  --output_dir /rscratch/data/steerable_pluralism/lora/${DATASET_PATH}/${STEERING_TYPE}/${MODEL_NICKNAME}/ \
--name ${DATASET_PATH}_${STEERING_TYPE}_${MODEL_NICKNAME} --dataset_path ${DATASET_PATH} --steering_type ${STEERING_TYPE} --model_name ${MODEL_NAME} \
--lr 8e-4 --num_epochs 10 --loss_function_type wd --which_scheduler='step' --warmup_ratio 0.10 --gamma 0.85

# cosine lr scheduler
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 \
--master_port=29507 recipes/quickstart/finetuning/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --use_peft --peft_method='lora' --use_fp16 --mixed_precision \
--batch_size_training 32 --val_batch_size 32 --gradient_accumulation_steps 1 \
--dist_checkpoint_root_folder ~/llama-recipes/outputs/llama2/ --dist_checkpoint_folder llama2 --batching_strategy='padding' \
--dataset opnqa_steering_dataset  --output_dir /rscratch/data/steerable_pluralism/lora/${DATASET_PATH}/${STEERING_TYPE}/${MODEL_NICKNAME}/ \
--name ${DATASET_PATH}_${STEERING_TYPE}_${MODEL_NICKNAME} --dataset_path ${DATASET_PATH} --steering_type ${STEERING_TYPE} --model_name ${MODEL_NAME} \
--lr 2e-4 --num_epochs 10 --loss_function_type wd --which_scheduler='cosine' --warmup_ratio 0.05 --gamma 0.85

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 \
--master_port=29507 recipes/quickstart/finetuning/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --use_peft --peft_method='lora' --use_fp16 --mixed_precision \
--batch_size_training 32 --val_batch_size 32 --gradient_accumulation_steps 1 \
--dist_checkpoint_root_folder ~/llama-recipes/outputs/llama2/ --dist_checkpoint_folder llama2 --batching_strategy='padding' \
--dataset opnqa_steering_dataset  --output_dir /rscratch/data/steerable_pluralism/lora/${DATASET_PATH}/${STEERING_TYPE}/${MODEL_NICKNAME}/ \
--name ${DATASET_PATH}_${STEERING_TYPE}_${MODEL_NICKNAME} --dataset_path ${DATASET_PATH} --steering_type ${STEERING_TYPE} --model_name ${MODEL_NAME} \
--lr 2e-4 --num_epochs 10 --loss_function_type wd --which_scheduler='cosine' --warmup_ratio 0.10 --gamma 0.85
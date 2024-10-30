export HF_TOKEN=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

cd ~/fork/llama-recipes/ 

export DATASET_PATH=leave_wave_34_out_remaining_9_1_0_split
export STEERING_TYPE=ALL

# export MODEL_NAME="/rscratch/data/llama-hf/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
# export MODEL_NICKNAME="Meta-Llama-2-7B"
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc-per-node=4 --master_port=29501 recipes/quickstart/finetuning/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --use_peft --peft_method='lora' --use_fp16 --mixed_precision --batch_size_training 32 --val_batch_size 32 --gradient_accumulation_steps 1 --dist_checkpoint_root_folder ~/llama-recipes/outputs/llama2/ --dist_checkpoint_folder llama2 --dataset opnqa_steering_dataset --batching_strategy='padding' --output_dir /rscratch/data/steerable_pluralism/lora/${DATASET_PATH}/${STEERING_TYPE}_${MODEL_NICKNAME} --name ${DATASET_PATH}_${STEERING_TYPE}_${MODEL_NICKNAME} --lr 4e-4 --num_epochs 10 --loss_function_type wd --dataset_path ${DATASET_PATH} --steering_type ${STEERING_TYPE} --model_name ${MODEL_NAME}

export MODEL_NAME="/rscratch/data/llama-hf/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"
export MODEL_NICKNAME="Meta-Llama-3-8B"
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc-per-node=4 --master_port=29501 recipes/quickstart/finetuning/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --use_peft --peft_method='lora' --use_fp16 --mixed_precision --batch_size_training 8 --val_batch_size 8 --gradient_accumulation_steps 4 --dist_checkpoint_root_folder ~/llama-recipes/outputs/llama2/ --dist_checkpoint_folder llama2 --dataset opnqa_steering_dataset --batching_strategy='padding' --output_dir /rscratch/data/steerable_pluralism/lora/${DATASET_PATH}/${STEERING_TYPE}_${MODEL_NICKNAME} --name ${DATASET_PATH}_${STEERING_TYPE}_${MODEL_NICKNAME} --lr 4e-4 --num_epochs 10 --loss_function_type wd --dataset_path ${DATASET_PATH} --steering_type ${STEERING_TYPE} --model_name ${MODEL_NAME}

export MODEL_NAME="/rscratch/data/llama-hf/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb"
export MODEL_NICKNAME="Meta-Llama-3p1-8B"
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc-per-node=4 --master_port=29501 recipes/quickstart/finetuning/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --use_peft --peft_method='lora' --use_fp16 --mixed_precision --batch_size_training 8 --val_batch_size 8 --gradient_accumulation_steps 4 --dist_checkpoint_root_folder ~/llama-recipes/outputs/llama2/ --dist_checkpoint_folder llama2 --dataset opnqa_steering_dataset --batching_strategy='padding' --output_dir /rscratch/data/steerable_pluralism/lora/${DATASET_PATH}/${STEERING_TYPE}_${MODEL_NICKNAME} --name ${DATASET_PATH}_${STEERING_TYPE}_${MODEL_NICKNAME} --lr 4e-4 --num_epochs 10 --loss_function_type wd --dataset_path ${DATASET_PATH} --steering_type ${STEERING_TYPE} --model_name ${MODEL_NAME}

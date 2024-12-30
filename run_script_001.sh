export HF_TOKEN=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

cd ~/fork/llama-recipes/ 

export DATASET_PATH=leave_wave_34_out_remaining_9_1_0_split
export MODEL_NAME="/rscratch/data/llama-hf/hub/trivia_calibrated_llama2_7b_pretrained"
export MODEL_NICKNAME="trivia_calibrated_llama2_7b_pretrained"

export STEERING_TYPE=QA
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc-per-node=4 --master_port=29501 recipes/quickstart/finetuning/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --use_peft --peft_method='lora' --use_fp16 --mixed_precision --batch_size_training 8 --val_batch_size 8 --gradient_accumulation_steps 4 --dist_checkpoint_root_folder ~/llama-recipes/outputs/llama2/ --dist_checkpoint_folder llama2 --dataset opnqa_steering_dataset --batching_strategy='padding' --output_dir /rscratch/data/steerable_pluralism/lora/${DATASET_PATH}/${STEERING_TYPE}_Meta-Llama-3-8B --name ${DATASET_PATH}_${STEERING_TYPE}_${MODEL_NICKNAME} --lr 4e-4 --num_epochs 10 --loss_function_type wd --dataset_path ${DATASET_PATH} --steering_type ${STEERING_TYPE} --model_name ${MODEL_NAME}

export STEERING_TYPE=BIO
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc-per-node=4 --master_port=29501 recipes/quickstart/finetuning/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --use_peft --peft_method='lora' --use_fp16 --mixed_precision --batch_size_training 8 --val_batch_size 8 --gradient_accumulation_steps 4 --dist_checkpoint_root_folder ~/llama-recipes/outputs/llama2/ --dist_checkpoint_folder llama2 --dataset opnqa_steering_dataset --batching_strategy='padding' --output_dir /rscratch/data/steerable_pluralism/lora/${DATASET_PATH}/${STEERING_TYPE}_Meta-Llama-3-8B --name ${DATASET_PATH}_${STEERING_TYPE}_${MODEL_NICKNAME} --lr 4e-4 --num_epochs 10 --loss_function_type wd --dataset_path ${DATASET_PATH} --steering_type ${STEERING_TYPE} --model_name ${MODEL_NAME}

export STEERING_TYPE=PORTRAY
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc-per-node=4 --master_port=29501 recipes/quickstart/finetuning/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --use_peft --peft_method='lora' --use_fp16 --mixed_precision --batch_size_training 8 --val_batch_size 8 --gradient_accumulation_steps 4 --dist_checkpoint_root_folder ~/llama-recipes/outputs/llama2/ --dist_checkpoint_folder llama2 --dataset opnqa_steering_dataset --batching_strategy='padding' --output_dir /rscratch/data/steerable_pluralism/lora/${DATASET_PATH}/${STEERING_TYPE}_Meta-Llama-3-8B --name ${DATASET_PATH}_${STEERING_TYPE}_${MODEL_NICKNAME} --lr 4e-4 --num_epochs 10 --loss_function_type wd --dataset_path ${DATASET_PATH} --steering_type ${STEERING_TYPE} --model_name ${MODEL_NAME}

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

## ADDED
from cycling_utils import TimestampedTimer
from cycling_utils import InterruptableDistributedSampler, MetricsTracker, AtomicDirectory

timer = TimestampedTimer()

from collections import Counter
import os

import dataclasses
import fire
import random
import torch
import torch.optim as optim
import torch.distributed as dist  
from peft import get_peft_model, PeftModel
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import quantization_config  as QUANTIZATION_CONFIG
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
    check_fsdp_config,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset,get_custom_data_collator

from llama_recipes.utils.fsdp_utils import hsdp_device_mesh
from llama_recipes.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    # print_model_size,
    get_policies,
    get_lr_scheduler,
)
from accelerate.utils import is_xpu_available
from warnings import warn, simplefilter

timer.report("imports")

simplefilter(action='ignore', category=FutureWarning)

def setup_wandb(train_config, fsdp_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama_recipes.configs import wandb_config as WANDB_CONFIG
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)
    return run

def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        # world_size = int(os.environ["WORLD_SIZE"]) # not used

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
    
    timer.report("init distributed environment")

    wandb_run = None

    if train_config.use_wandb:
        if not train_config.enable_fsdp or rank==0:
            wandb_run = setup_wandb(train_config, fsdp_config, **kwargs)
        timer.report("init wandb")
    
    #setting quantization configs
    bnb_config = None
    if train_config.quantization:
        if type(train_config.quantization) == type(True):
            warn("Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.", FutureWarning)
            train_config.quantization = "8bit"

        if train_config.quantization == "8bit" and train_config.enable_fsdp:
            raise ValueError("8bit quantization is not supported with FSDP, please use 4bit quantization")

        quant_config = QUANTIZATION_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(train_config.quantization)
        timer.report("init quantization")

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    timer.report(f"use_cache: {use_cache}")

    config = AutoConfig.from_pretrained(train_config.model_name)
    if config.model_type == "llama":
        timer.report(f"loading LlamaForCausalLM.from_pretrained")
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            quantization_config=bnb_config,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
            torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
        )
        timer.report("loaded LlamaForCausalLM.from_pretrained")
    else:
        raise ValueError(f"Model type {config.model_type} is not supported. Please use llama or mllama model.")

    # Load the tokenizer and add special tokens
    timer.report("Load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name)
    if not tokenizer.pad_token_id: 
        tokenizer.pad_token_id = tokenizer.eos_token_id
    timer.report("loaded tokenizer")
        
    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16 and not train_config.quantization:
        timer.report("model to torch.bfloat16")
        model.to(torch.bfloat16)
        timer.report("Finished model to torch.bfloat16")
        
    if train_config.use_peft:
        latest_sym_path = os.path.join(kwargs["dist_checkpoint_root_folder"], "latest_pt")
        if os.path.exists(latest_sym_path):
            latest_path = os.readlink(latest_sym_path)
            peft_path = os.path.join(latest_path, "peft")
            dist.barrier()
            model = PeftModel.from_pretrained(model, peft_path, is_trainable=True)
            if rank == 0:
                print("RESUMED SAVED PEFT MODULES")
            dist.barrier()
        # Generate the peft config and start fine-tuning from original model
        else:
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(model, peft_config)
            if rank == 0:
                print("BUILT NEW PEFT MODULES")
            if wandb_run:
                wandb_run.config.update(peft_config)
        if rank == 0:
            model.print_trainable_parameters()
        timer.report("peft modules inserted")

    hsdp_device_mesh_plan = None
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh_plan = hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size, sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        check_fsdp_config(fsdp_config)
        
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(model, train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        # Create the FSDP wrapper for MllamaSelfAttentionDecoderLayer,MllamaSelfAttentionDecoderLayer,MllamaVisionEncoderLayer in vision models
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [LlamaDecoderLayer])
        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()
        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh_plan,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=(lambda module: module.to_empty(device=torch.device("cuda"), recurse=False))
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        timer.report("FSDP wrapping done")
        if fsdp_config.fsdp_activation_checkpointing:            
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            apply_fsdp_checkpointing(model)
            timer.report("FSDP activation checkpointing enabled")                      
    elif not train_config.quantization and not train_config.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")
    dataset_config = generate_dataset_config(train_config, kwargs)
    dataset_processer = tokenizer

    timer.report("FSDP applied")

    # Load and preprocess the dataset for training and validation

    dataset_train = get_preprocessed_dataset(
        dataset_processer,
        dataset_config,
        split="train",
    )
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        dataset_processer,
        dataset_config,
        split="valid",
    )

    dataset_test = get_preprocessed_dataset(
        dataset_processer,
        dataset_config,
        split="test",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(dataset_val)}")

    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, dataset_processer, "train")
    print("length of dataset_train", len(dataset_train))

    ## -- INSERT INTERRUPTIBLE DISTRIBUTED SAMPLER -- ##
    train_sampler = InterruptableDistributedSampler(dataset_train)
    train_dl_kwargs["sampler"] = train_sampler

    custom_data_collator = get_custom_data_collator(dataset_processer,dataset_config)
    if custom_data_collator:
        print("custom_data_collator is used")
        train_dl_kwargs["collate_fn"] = custom_data_collator
    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )
    # compute the total number of steps for training (num_epochs * len(train_dataloader))
    if train_config.max_train_step > 0:
        train_config.num_epochs = train_config.max_train_step // len(train_dataloader) + 1
    else:
        train_config.max_train_step = train_config.num_epochs * len(train_dataloader)

    print(f"--> Num of Training Set Batches loaded = {len(train_dataloader)}")

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, dataset_processer, "val")

        ## -- INSERT INTERRUPTIBLE DISTRIBUTED SAMPLER -- ##
        val_sampler = InterruptableDistributedSampler(dataset_val)
        val_dl_kwargs["sampler"] = val_sampler

        if custom_data_collator:
            val_dl_kwargs["collate_fn"] = custom_data_collator
    
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
        print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")
        if len(eval_dataloader) == 0:
            raise ValueError("The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set.")
        else:
            print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")

    test_dataloader = None
    if train_config.run_test:
        test_dl_kwargs = get_dataloader_kwargs(train_config, dataset_test, dataset_processer, "test")
        if custom_data_collator:
            test_dl_kwargs["collate_fn"] = custom_data_collator

        test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **test_dl_kwargs,
        )
        print(f"--> Num of Test Set Batches loaded = {len(test_dataloader)}")
        if len(test_dataloader) == 0:
            raise ValueError("The test set size is too small for dataloader to load even one batch. Please increase the size of test set.")
        else:
            print(f"--> Num of Test Set Batches loaded = {len(test_dataloader)}")
    timer.report("datasets and dataloaders")
    
    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr= train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    lr_scheduler, update_freq = get_lr_scheduler(optimizer, train_config)

    ## -- MOVED OUTSIDE TRAINING LOOP -- ##
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.amp.GradScaler()
    
    # Init metric trackerb and saver
    metrics = {'train': MetricsTracker(), 'eval': MetricsTracker()}
    saver = AtomicDirectory(kwargs["dist_checkpoint_root_folder"])

    timer.report("optimizer, scaler, metrics, and saver")

    ## -- LOAD SAVED OPTIMIZER AND SCHEDULER IF SAVED PREVIOUSLY -- ##
    start_epoch = 0

    latest_sym_path = os.path.join(kwargs["dist_checkpoint_root_folder"], saver.symlink_name)
    if os.path.exists(latest_sym_path):
        latest_path = os.readlink(latest_sym_path)
        peft_path = os.path.join(latest_path, "peft")
        osd_path = os.path.join(latest_path, "optimizer.pt")
        other_checkpoint_path = os.path.join(latest_path, "other_checkpoint.pt")

        if rank == 0:
            print("RESUMING FROM CHECKPOINTS")

        full_osd = None
        if rank == 0:
            full_osd = torch.load(osd_path)
        sharded_osd = FSDP.scatter_full_optim_state_dict(full_optim_state_dict=full_osd, model=model, optim=optimizer)
        optimizer.load_state_dict(sharded_osd)

        checkpoint = torch.load(other_checkpoint_path)       
        scaler.load_state_dict(checkpoint["scaler"])
        train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
        eval_dataloader.sampler.load_state_dict(checkpoint["eval_sampler"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        metrics = checkpoint["metrics"]
        start_epoch = train_dataloader.sampler.epoch

    timer.report("checkpoint retrieval")

    ## -- MOVE TRAINING LOOP OUTSIDE TRAIN FUNCTION -- ##
    for epoch in range(start_epoch, train_config.num_epochs):
        if rank == 0:
            print(f"\nEPOCH :: {epoch + 1}\n")
        
        with train_dataloader.sampler.in_epoch(epoch):
            # Start the training process
            train(
                epoch,
                model,
                train_dataloader,
                eval_dataloader,
                optimizer,
                lr_scheduler,
                scaler, ## ADDED
                metrics, ## ADDED
                timer, ## ADDED
                train_config.gradient_accumulation_steps,
                train_config,
                saver ## ADDED
            )

if __name__ == "__main__":
    fire.Fire(main)

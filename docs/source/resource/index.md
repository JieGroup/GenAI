# 6. Efficiency and Memory in Large Model Training


## Accelerate Large Model Training using DeepSpeed

Training large models often leads to OOM errors due to hardware limitations. Data Parallelism using ZeRO can help optimize the use of available hardware by distributing optimizer states, gradients, and model parameters across multiple GPUs. This post explores how to leverage DeepSpeed ZeRO using Accelerate.

This part explains how to use the Accelerate library for training large models, leveraging the Zero Redundancy Optimizer (ZeRO) features of DeepSpeed. It addresses common issues such as Out of Memory (OOM) errors and demonstrates how to efficiently utilize hardware for large model training. Much of the content is borrowed from [this blog](https://huggingface.co/blog/accelerate-deepspeed).

### ZeRO Data Parallelism

The ZeRO optimizer has three stages, each progressively offloading more data to optimize memory usage:

1. **Stage 1**: Shards optimizer states across data parallel workers/GPUs.
2. **Stage 2**: Shards optimizer states and gradients across data parallel workers/GPUs.
3. **Stage 3**: Shards optimizer states, gradients, and model parameters across data parallel workers/GPUs.
4. **Optimizer Offload**: Offloads gradients and optimizer states to CPU/Disk (building on Stage 2).
5. **Param Offload**: Offloads model parameters to CPU/Disk (building on Stage 3).

 
Below is a short description of Data Parallelism using ZeRO with diagram from this [blog post](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/): ![ZeRO Data Parallelism](https://www.microsoft.com/en-us/research/uploads/prod/2020/02/DeepSpeed-Image-1.png) *Figure 1: Memory savings and communication volume for the three stages of ZeRO compared with standard data parallel baseline. In the memory consumption formula, Ψ refers to the number of parameters in a model and K is the optimizer specific constant term. As a specific example, we show the memory consumption for a 7.5B parameter model using [Adam](https://arxiv.org/pdf/1412.6980.pdf) optimizer where K=12 on 64 GPUs. We also show the communication volume of ZeRO relative to the baseline.*


The video below shows how ZeRO (with all three stages) performs a training step including forward pass, backward pass, and parameter update: <video controls> <source src="https://www.microsoft.com/en-us/research/uploads/prod/2020/02/Turing-Animation.mp4?_=1" type="video/mp4"> Your browser does not support the video tag. </video>

### Using Accelerate with DeepSpeed ZeRO

#### Hardware Setup

- **GPUs**: 2x24GB NVIDIA Titan RTX
- **RAM**: 60GB

#### Example 1: Finetuning a Sequence-to-Sequence Model for Chatbot Training

We will finetune `facebook/blenderbot-400M-distill` on the MuDoConv dataset.

#### Configuration File

Create a `zero2_config_accelerate.json` file:
 
```json
{
    "fp16": {
        "enabled": "true",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 15,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "weight_decay": "auto",
            "torch_adam": true,
            "adam_w_mode": true
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

To enable DeepSpeed ZeRO Stage-2 with above config, please run  `accelerate config`  and provide the config file path when asked. For more details, refer the `accelerate`  official documentation for  [DeepSpeed Config File](https://huggingface.co/docs/accelerate/deepspeed#deepspeed-config-file).
The code is available here [run_seq2seq_no_trainer.py](https://github.com/pacman100/accelerate-deepspeed-test/blob/main/src/modeling/run_seq2seq_no_trainer.py)

**ZeRO Stage-2 DeepSpeed Config File Example**

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
 deepspeed_config_file: /path/to/zero2_config_accelerate.json
 zero3_init_flag: false
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

#### Training Command

Now, run below command for training:
```python
accelerate launch run_seq2seq_no_trainer.py \
    --dataset_name "smangrul/MuDoConv" \
    --max_source_length 128 \
    --source_prefix "chatbot: " \
    --max_target_length 64 \
    --val_max_target_length 64 \
    --val_min_target_length 20 \
    --n_val_batch_generations 5 \
    --n_train 10000 \
    --n_val 1000 \
    --pad_to_max_length \
    --num_beams 10 \
    --model_name_or_path "facebook/blenderbot-400M-distill" \
    --per_device_train_batch_size 200 \
    --per_device_eval_batch_size 100 \
    --learning_rate 1e-6 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 100 \
    --output_dir "/tmp/deepspeed_zero_stage2_accelerate_test" \
    --seed 25 \
    --logging_steps 100 \
    --with_tracking \
    --report_to "wandb" \
    --report_name "blenderbot_400M_finetuning"
```

#### Results

| Method                      | Batch Size Max | Eval Size Max | Train time per epoch (seconds) | Eval time per epoch (seconds) |
|-----------------------------|----------------|---------------|-------------------------------|-------------------------------|
| DDP (Distributed Data Parallel) | 100            | 50            | 27.36                         | 48.41                         |
| DeepSpeed ZeRO Stage 2      | 200            | 100           | 19.06                         | 39.27                         |

*Table 2: Benchmarking DeepSpeed ZeRO Stage-2 on BlenderBot (400M) model*


As this model is of medium size, the speedup isn't that exciting but this will improve with bigger models.

### CPU/Disk Offloading with DeepSpeed ZeRO Stage-3

#### Example 2: Training GPT-XL Model (1.5B parameters)

On a single 24GB NVIDIA Titan RTX GPU, one cannot train GPT-XL Model (1.5B parameters) even with a batch size of 1. We will look at how we can use DeepSpeed ZeRO Stage-3 with CPU offloading of optimizer states, gradients and parameters to train GPT-XL Model.

Create a `zero3_offload_config_accelerate.json` file:

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "
```

**ZeRO Stage-3 CPU Offload DeepSpeed Config File Example**

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
 deepspeed_config_file: /path/to/zero3_offload_config_accelerate.json
 zero3_init_flag: true
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

Now, run below command for training:

```python
accelerate launch run_clm_no_trainer.py \
--config_name "gpt2-xl" \
--tokenizer_name "gpt2-xl" \
--dataset_name "wikitext" \
--dataset_config_name "wikitext-2-raw-v1" \
--block_size 128 \
--output_dir "/tmp/clm_deepspeed_stage3_offload__accelerate" \
--learning_rate 5e-4 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 1 \
--num_train_epochs 1 \
--with_tracking \
--report_to "wandb"\
```


The following result shows that DDP will result in OOM error even with batch size 1. On the other hand, with DeepSpeed ZeRO Stage-3 CPU offload, we can train with a batch size of 16.

| Method                      | Batch Size Max | Train time per epoch (seconds) | Notes       |
|-----------------------------|----------------|-------------------------------|-------------|
| DDP (Distributed Data Parallel) | -              | -                             | OOM Error   |
| DeepSpeed ZeRO Stage 3      | 16             | 6608.35                        |             |

*Table 3: Benchmarking DeepSpeed ZeRO Stage-3 CPU Offload on GPT-XL (1.5B) model*



## From DeepSpeed to FSDP

DeepSpeed and PyTorch's Fully Sharded Data Parallel (FSDP) are two popular implementations of the ZeRO Redundancy Optimizer (Zero) algorithm. Both frameworks are designed to optimize the training of large models by reducing memory redundancy and improving computational efficiency. This guide highlights their similarities, differences, and how they can be utilized effectively in various scenarios.


### Differences Between DeepSpeed and FSDP

 - Parameter Handling

	- **DeepSpeed**: Internally upcasts parameters to float32 during training, which can impact memory usage and precision.
	- **FSDP**: Operates with the precision specified by the user (e.g., bfloat16), offering more flexibility in managing memory and precision.

- Optimizer Initialization

	- **DeepSpeed**: Initializes optimizer parameters in float32 regardless of the specified training precision.
	- **FSDP**: Initializes optimizer parameters according to the specified precision, allowing for more efficient memory usage in low-precision training scenarios.

- Mixed Precision

	- **DeepSpeed**: Uses a fixed mixed precision strategy defined in its configuration files.
	- **FSDP**: Adapts to the specified precision, allowing users to configure mixed precision training more flexibly.

#### Precision Configuration Comparison

| Framework | Model Loading Precision | Mixed Precision | Local Preparation | Training | Local Optimizer |
|-----------|--------------------------|-----------------|-------------------|----------|-----------------|
| FSDP (memory-constrained) | bfloat16 | None (default)  | bfloat16          | bfloat16 | bfloat16        |
| FSDP (mixed precision)    | bfloat16 | bfloat16        | float32           | bfloat16 | float32         |
| DeepSpeed                 | bfloat16 | bfloat16        | float32           | bfloat16 | float32         |



In summary, DeepSpeed maintains master weights in float32 by default, performing internal upcasting. This approach can lead to higher memory consumption but ensures stability in optimizer convergence. FSDP allows users to specify the training precision (e.g., bfloat16), which can result in lower memory usage and potentially faster training times in memory-constrained environments. Hugging Face Accelerate package supports both DeepSpeed and FSDP.  

### References

- [ZeRO Redundancy Optimizer (Zero)](https://arxiv.org/abs/1910.02054)
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/en/index)
- [FSDP and DeepSpeed](https://huggingface.co/blog/deepspeed-to-fsdp-and-back)

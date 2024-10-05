import os
import random
import string

def random_string(length=8):
    return ''.join(random.choice(string.ascii_letters + string.digits) for i in range(length))


if __name__ == "__main__":

    '''
        make parallel jobs
    '''
    commands = [
        # finetuning Llama2 using all 52k data
        'python finetune.py \
            --base_model "meta-llama/Llama-2-7b-hf" \
            --load_8bit True \
            --load_4bit False \
            --data_path "yahma/alpaca-cleaned" \
            --output_dir "./lora-alpaca-8bit" \
            --batch_size 128 \
            --micro_batch_size 16 \
            --num_epochs 3\
            --learning_rate 1e-4 \
            --cutoff_len 512 \
            --val_set_size 2000 \
            --lora_r 8 \
            --lora_alpha 16 \
            --lora_dropout 0.05 \
            --lora_target_modules "[q_proj,v_proj]" \
            --train_on_inputs \
            --group_by_length \
            --wandb_run_name "instruct-lora8bit-llama2-7B"',
        # finetuning using random 1k data
        'python finetune_Jie.py \
            --base_model "meta-llama/Llama-2-7b-hf" \
            --load_8bit True \
            --load_4bit False \
            --data_path "yahma/alpaca-cleaned" \
            --training_size 1000 \
            --output_dir "./lora-alpaca-8bit-trainsize1000" \
            --batch_size 128 \
            --micro_batch_size 16 \
            --num_epochs 1 \
            --learning_rate 1e-4 \
            --cutoff_len 512 \
            --val_set_size 200 \
            --lora_r 2 \
            --lora_alpha 16 \
            --lora_dropout 0.05 \
            --lora_target_modules "[q_proj,v_proj]"',
    ]

    template_path = 'main.pbs'
    for idx, command in enumerate(commands):
        # Create a new PBS file for each command
        with open(template_path, 'r') as template_file:
            content = template_file.read()
            content = content.replace("COMMAND_PLACEHOLDER", command)

        job_file_name = f'job_{idx}_{random_string()}.pbs'
        with open(job_file_name, 'w') as job_file:
            job_file.write(content)
        # print(f'created sbatch {job_file_name} into {bash_file_name}')
        
        # Submit the job
        os.system(f'sbatch {job_file_name}')

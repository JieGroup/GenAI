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
#         'python finetune.py \
#             --base_model "/home/aanwar/wang8740/llama/model_output/llama-2-7b" \
#             --load_8bit True \
#             --load_4bit False \
#             --data_path "yahma/alpaca-cleaned" \
#             --output_dir "./lora-alpaca-8bit" \
#             --batch_size 128 \
#             --micro_batch_size 16 \
#             --num_epochs 3\
#             --learning_rate 1e-4 \
#             --cutoff_len 512 \
#             --val_set_size 2000 \
#             --lora_r 8 \
#             --lora_alpha 16 \
#             --lora_dropout 0.05 \
#             --lora_target_modules "[q_proj,v_proj]" \
#             --train_on_inputs \
#             --group_by_length \
#             --resume_from_checkpoint "/home/aanwar/wang8740/recommender/lora-alpaca-8bit/checkpoint-1000" \
#             --wandb_run_name "instruct-lora8bit-llama2-7B"',
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
            --lora_target_modules "[q_proj,v_proj]"', # \
            # --train_on_inputs \
            # --group_by_length \
            # --wandb_run_name "instruct-lora8bit-llama2-7B-trainsize1000"',
        # finetuning using only 4bit + Lora 
#         'python finetune.py \
#             --base_model "/home/aanwar/wang8740/llama/model_output/llama-2-7b" \
#             --load_8bit False \
#             --load_4bit True \
#             --data_path "yahma/alpaca-cleaned" \
#             --output_dir "./lora-alpaca-4bit" \
#             --batch_size 128 \
#             --micro_batch_size 16 \
#             --num_epochs 1 \
#             --learning_rate 1e-4 \
#             --cutoff_len 512 \
#             --val_set_size 2000 \
#             --lora_r 8 \
#             --lora_alpha 16 \
#             --lora_dropout 0.05 \
#             --lora_target_modules "[q_proj,v_proj]" \
#             --train_on_inputs \
#             --group_by_length \
#             --resume_from_checkpoint "/home/aanwar/wang8740/recommender/lora-alpaca-4bit/checkpoint-400" \
#             --wandb_run_name "instruct-lora4bit-llama2-7B"',
        # merge the lora adaptor with the original llama2
#         'python merge_lora.py \
#             --base_model "/home/aanwar/wang8740/llama/model_output/llama-2-7b" \
#             --lora_adapter_dir "./lora-alpaca-8bit" \
#             --merged_model_dir "/home/aanwar/wang8740/recommender/result/our-instruct-llama2"'
        # finetuning the rec data using original Llama2 -- only for test
#         'python finetune_rec.py \
#              --base_model "/home/aanwar/wang8740/llama/model_output/llama-2-7b" \
#              --train_data_path "/home/aanwar/wang8740/recommender/data/movie/train.json" \
#              --val_data_path "/home/aanwar/wang8740/recommender/data/movie/valid.json" \
#              --training_size 64 \
#              --output_dir "/home/aanwar/wang8740/recommender/result/finetune_rec_64" \
#              --batch_size 12 \
#              --micro_batch_size 6 \
#              --num_epochs 10 \
#              --learning_rate 1e-4 \
#              --cutoff_len 512 \
#              --lora_r 8 \
#              --lora_alpha 16\
#              --lora_dropout 0.05 \
#              --lora_target_modules "[q_proj,v_proj]" \
#              --train_on_inputs \
#              --group_by_length',
        # finetuning the rec data using our Llama2 finetuned from 52k
#         'python finetune_rec.py \
#              --base_model "/home/aanwar/wang8740/recommender/result/our-instruct-llama2" \
#              --train_data_path "/home/aanwar/wang8740/recommender/data/movie/train.json" \
#              --val_data_path "/home/aanwar/wang8740/recommender/data/movie/valid.json" \
#              --training_size 256 \
#              --output_dir "/home/aanwar/wang8740/recommender/result/finetune_rec256_our-instruct-llama2" \
#              --batch_size 12 \
#              --micro_batch_size 6 \
#              --num_epochs 10 \
#              --learning_rate 1e-4 \
#              --cutoff_len 512 \
#              --lora_r 8 \
#              --lora_alpha 16\
#              --lora_dropout 0.05 \
#              --lora_target_modules "[q_proj,v_proj]" \
#              --train_on_inputs \
#              --group_by_length',
        #
        # 'python evaluate_rec.py \
        #     --base_model "/home/aanwar/wang8740/recommender/result/our-instruct-llama2" \
        #     --lora_weights "/home/aanwar/wang8740/recommender/result/finetune_rec64_our-instruct-llama2" \
        #     --train_data_path "/home/aanwar/wang8740/recommender/data/movie/train.json" \
        #     --test_data_path "/home/aanwar/wang8740/recommender/data/movie/valid.json" \
        #     --dataset_name "movie" \
        #     --training_size 64 \
        #     --result_json_data "result/finetune_rec.json" \
        #     --batch_size 8',
#         'python evaluate_rec.py \
#             --base_model "/home/aanwar/wang8740/recommender/result/our-instruct-llama2" \
#             --lora_weights "/home/aanwar/wang8740/recommender/result/finetune_rec128_our-instruct-llama2" \
#             --train_data_path "/home/aanwar/wang8740/recommender/data/movie/train.json" \
#             --test_data_path "/home/aanwar/wang8740/recommender/data/movie/valid.json" \
#             --dataset_name "movie" \
#             --training_size 128 \
#             --result_json_data "result/finetune_rec2.json" \
#             --batch_size 8',
#         'python evaluate_rec.py \
#             --base_model "/home/aanwar/wang8740/recommender/result/our-instruct-llama2" \
#             --lora_weights "/home/aanwar/wang8740/recommender/result/finetune_rec256_our-instruct-llama2" \
#             --train_data_path "/home/aanwar/wang8740/recommender/data/movie/train.json" \
#             --test_data_path "/home/aanwar/wang8740/recommender/data/movie/valid.json" \
#             --dataset_name "movie" \
#             --training_size 256 \
#             --result_json_data "result/finetune_rec3.json" \
#             --batch_size 8',
#         'python generate.py \
#             --load_8bit \
#             --base_model "/home/aanwar/wang8740/llama/model_output/llama-2-7b" \
#             --lora_weights "/home/aanwar/wang8740/recommender/lora-alpaca-8bit"'
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

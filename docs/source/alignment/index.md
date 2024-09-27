# 4. Fine-Tuning LLMs
<!-- and Human Value Alignment -->


## Overview of Using LLM in Practice

```{mermaid}
graph LR
    UseCase[Define Science Domain] --> DataCuration[Data Curation]
    DataCuration --> Architecture[Set Architecture]
    Architecture --> Training[Training]
    Training --> Base[Base Model]
```
```{mermaid}
graph LR
    BaseModel[Base Model] --> FineTuning[Supervised Fine-tuning]
	UseCase[Define Application Case] --> FineTuning[Supervised Fine-tuning]
    FineTuning --> RLHF[Human Value Alignment]
```

-   **Base Model**: Typically involves training a transformer-based model on a diverse dataset to learn a broad representation of language (which is not necessarily human language). 
	> This can be likened to a well-educated individual who has a broad understanding of various topics but has not specialized in any. They have potential because they are well-rounded and knowledgeable, but lack specific skills or experiences.
    
-   **Supervised Fine-tuning**: After the base training, the model is fine-tuned on more specific datasets. This step helps the model adapt to particular tasks or domains by learning from labeled data that provide direct examples of desired outputs.
	> This process can be compared to job-specific training where the individual applies their broad knowledge to a particular domain. They become more adept at handling specific tasks relevant to that domain, thus gaining depth in addition to their breadth.
    
-   **Human Value Alignment**: This step often the techniques of Reinforcement Learning from Human Feedback (RLHF) to further finetune the model to refine its responses based on human feedback. It is used to align the model's outputs more closely with human values and preferences, enhancing its applicability in practical scenarios.
	> This step could be likened to personalized coaching or mentoring, where the individual refines their skills further, focusing on particular nuances and preferences that are highly valued in specific contexts or by particular users.

We will go through model finetuning in this chapter and defer the human-value alignment issues in the next chapter. 



## Fine-Tuning Techniques

Fine-tuning is the process of taking a pre-trained model and adapting it to a specific task or domain. This approach leverages the general language understanding that the model has already acquired during its initial training on large-scale text corpora. By fine-tuning, we specialize the model for tasks such as text classification, question-answering, and multi-round conversations.
Fine-tuning techniques can be categorized into two main approaches: Full-parameter Fine-Tuning and Parameter-efficient Fine-Tuning (including Transfer Learning).



### Full-parameter Fine-Tuning 
All parameters of the base model are updated during training:

$$
\theta = \theta − \eta \cdot \nabla_{\theta} L(\theta).
$$ 

This is the most comprehensive way to adapt a model to a new task or domain. However, it requires substantial computational resources and time, and could cause overfitting on small datasets.

### Parameter-efficient Fine-Tuning (PEFT)

PEFT addresses the need to fine-tune large pretrained models efficiently, especially when computational resources are constrained. Instead of updating all parameters in the base model, PEFT methods selectively **update a small subset of parameters** or **introduce additional trainable components** while freezing most of the pretrained network. This approach maintains the efficiency of transfer learning without the computational cost of full-parameter fine-tuning.

$$
\theta_{task} = \theta_{task} − \eta \cdot \nabla_{\theta} L(\theta_{task}, \theta_{frozen}).
$$ 


Several PEFT methods have been proposed, each offering unique ways to fine-tune the model:

#### Adapter Layers
Adapters are small bottleneck layers inserted between the layers of the pretrained model. During fine-tuning, only the parameters of these adapters are updated, while the rest of the model is frozen. However, adapters introduce additional sequential computations that cause inference latency.

#### Prefix Tuning
It introduces continuous vectors, known as the prefix, which are prepended to either the input embeddings or hidden states of the model. Specifically, the modified hidden state $h_{\text{modified}}$ is obtained by concatenating a prefix vector $P_{\text{prefix}}$ with the original hidden state $h$:

$$
h_{\text{modified}} = \text{Concat}(P_{\text{prefix}}, h)
$$

where \( P_{\text{prefix}} \) is learned during fine-tuning. These tunable vectors serve as prompts that guide the model’s behavior without altering the original model weights. The advantage is that it allows parallel computation of the modified hidden representations. However, it has been noted to suffer from optimization difficulties and lack of stability during training.

#### Low-Rank Adaptation (LoRA)
LoRA introduces low-rank matrices that reparameterize the updates of pretrained weight matrices. The fine-tuning happens in a low-dimensional subspace. Specifically, for a pre-trained large weight matrix $W_0 \in \mathbb{R}^{m \times n}$, let $\Delta W$ be its update during the fine-tuning, that is, the updated weighted matrix is $W_0+\Delta W$.
LoRA constrains each update to have a low-rank representation:

$$
\Delta W = \alpha BA,
$$

where $B \in \mathbb{R}^{m \times r}$ and $A \in \mathbb{R}^{r \times n}$ are low-rank matrices with rank $r \ll \min(m, n)$, and $\alpha > 0$ is a scaling factor. During the entire training stage, the pre-trained weights $W_0$ are fixed while $A$ and $B$ are trainable parameters, thus requiring fewer trainable parameters. 
To produce predictions during the inference stage, the contribution of the low-rank matrices can be integrated into the updated weight matrix: 

$$
W' = W_0 + \alpha BA.
$$

LoRA clearly has the advantage in lightweight fine-tuning scenarios. However, the low-rank updates might not capture the complexity required for highly specialized or difficult tasks over long epochs.


#### Parameter-free Fine-tuning 
Gradient Learning (GL) is a recent approach to be both parameter-free and model-agnostic. The key concept behind GL is Gradient Decoupling, which separates the computation of gradients for auxiliary parameters from the fine-tuned hidden representations of the model. Specifically, inspired by the principles of Gradient Boosting, which optimizes models iteratively through functional gradients, GL applies a similar idea to neural networks by offloading gradient computations to lower-cost devices, such as CPUs or low-end GPUs. 



$$
\text{Parameter-based fine-tuning: } 
\nabla_{\theta_{1:M}} L\left(f_{\theta_{1:M}}\right)
$$

$$
\text{Parameter-free fine-tuning: } 
\nabla_{\hat{h}_{1:M}} L(f_\theta(\Delta h_{1:M}))
$$


GL offers flexibility by working with any model for adapter-based training and offloading computations to lower-cost devices, reducing the load on expensive resources. However, it can lead to increased training latency due to the need for gradient offloading across different devices.



## Instruction Fine-Tuning
An important example of fine-tuning LLMs is instruction finetuning. It aims to make a language model follow instructions more closely by fine-tuning on a dataset of instruction-following examples. 

A typical Fine-tuning involves the following steps:

- Data Preparation: Gather and preprocess a dataset that is representative of the task you want the model to perform. For instruction fine-tuning, this data typically includes input-output pairs where the model is given specific prompts and expected responses.

- Model Adaptation: Modify the architecture or specific components of the pre-trained model if necessary, to better suit the fine-tuning task.

- Training: Train the model on the task-specific dataset using a smaller learning rate to prevent catastrophic forgetting of the pre-trained knowledge.

Here is an example code for reproducing Alpaca, an instruction-finetuned model based on the original LLaMA. The difference is that we are fine-tuning on LLaMA2-7B.

```bash
pip install peft wandb datasets
pip install -U bitsandbytes
```

Download the utility Python module for this chapter [here](https://drive.google.com/file/d/1g7mTV5HMxYstud-8Q2wIegedo_AnoUhb/view?usp=sharing).
```python
import os
from typing import List
import fire
import torch
import transformers
from datasets import load_dataset
import random
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training,
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from utils_finetune import Prompter

def train(
    # model/data params
    base_model: str = "",  # The only required argument
    load_8bit: bool = True,
    load_4bit: bool = False,
    data_path: str = "yahma/alpaca-cleaned",
    training_size: int = -1, # Default: using all the training sample
    output_dir: str = "results/instruction-llama2-7B",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False, # append an "End of Sequence" (EOS) token
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):

    gradient_accumulation_steps = batch_size // micro_batch_size
    prompter = Prompter(prompt_template_name)

    # quantized training, where model weights are compressed to save memory and computational power
    if load_8bit: quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    if load_4bit: quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = LlamaForCausalLM.from_pretrained(base_model, quantization_config=quantization_config, torch_dtype=torch.float16, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = (0) # unk. set to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
        if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len and add_eos_token:
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(data_point["instruction"], data_point["input"], data_point["output"])
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(data_point["instruction"], data_point["input"])
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            if add_eos_token: user_prompt_len -= 1
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True)
        print(f'Training size is {len(train_val["train"])}')
        if training_size > -1:
            sampled_indices = random.sample(range(len(train_val["train"])), training_size)
            train_data = train_val["train"].select(sampled_indices).shuffle().map(generate_and_tokenize_prompt)
        else:
            train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
 
    # Fix the loading issue at https://github.com/tloen/alpaca-lora/issues/319#issuecomment-1780470884
    # This callback ensures that only the PEFT model (LoRA layers) is saved during checkpoints instead of the entire base model
    class SavePeftModelCallback(TrainerCallback):
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)
            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control
        
    trainer = transformers.Trainer(
        model=peft_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            group_by_length=group_by_length,
        ),
        # A utility that ensures the input sequences are padded to the correct length to make the tensor dimensions uniform across batches
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=[SavePeftModelCallback],
    )
    peft_model.config.use_cache = False
    trainer.train()
    peft_model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)
```

Use a particular set of parameters and a sub dataset of size 1000 to train the model. 
```python
python finetune.py \
    --base_model "/home/aanwar/wang8740/llama/model_output/llama-2-7b" \
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
    --lora_target_modules "[q_proj,v_proj]"
```

## Application Case: LLM-based Recommender Systems

In this section, we will explore building a recommender system based on LLMs fine-tuned on a dataset of user preferences and unpreferences in books and movies. The goal is to fine-tune a model that can predict whether a user will like a specific book or movie based on a few existing ratings they have provided.

### Dataset Overview
The `book_movie_data` dataset is available with fileID `1zbYfV18dMEpebJjnaaIDSiiTeDBd07sI`. 
The dataset contains user preferences for books and movies. It is organized into two main folders, /book and /movie, each containing train.json, test.json, and valid.json files. Each entry in these files includes the following fields:

- instruction: A prompt asking the model to predict whether a user will like a target item based on their preferences and unpreferences.
- input: A list of user-preferred and unpreferred books or movies.
- output: The answer is either "Yes" or "No", indicating whether the user is likely to enjoy the target item.

For example, the dataset might contain the following structure:

```
{
    "instruction": "Given the user's preference and unpreference, identify whether the user will like the target book by answering \"Yes.\" or \"No.\".",
    "input": "User Preference: \"The Bean Trees\" written by Barbara Kingsolver, \"Sula\" written by Toni Morrison\nUser Unpreference: \nWhether the user will like the target book \"Epitaph for a Peach\" written by David M. Masumoto?",
    "output": "No."
}
```

### Pipeline Overview

Consider a pipeline that consists of the following steps:

- Data Preparation: The data is ready.
- Model Preparation: Initialize a pre-trained LLM and fine-tune it using PEFT techniques
- Training: Fine-tune and use early stopping to alleviate overfitting.
- Evaluation: Consider metrics such as AUC (Area Under the Curve) to measure how well the model distinguishes between positive and negative predictions
- Inference: Once trained, the model can generate personalized recommendations based on a user's preference history. Try it out and see what it recommends for youself!



### Sample code

For an easy start, we provide the following sample code snippets that you may want to use as a reference for structuring your training loop.

```python
def compute_metrics(eval_preds):
    probs, labels = eval_preds[0]
    auc = roc_auc_score(labels, probs)
    return {'auc': auc}

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    ID_yes, ID_no = 8241, 3782
    labels_index = torch.argwhere(torch.bitwise_or(labels == ID_yes, labels == ID_no)) 
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == ID_no, 0, 1)
    labels_index[: , 1] = labels_index[: , 1] - 1
    logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:,[ID_no, ID_yes]], dim = -1)
    return logits[:, 1][2::3], gold[2::3]

trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        ...
    ),
    data_collator=...
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=10), SavePeftModelCallback]
)
```

## Application Case: Fine-tuning LLM for Tabular Data

### Motivation
The application of LLMs for tabular data prediction opens up new possibilities for data analysis, allowing models to leverage contextual information within the data, which traditional models might overlook. 

### Traditional Problems in Tabular Data Analysis

Consider a standard classification problem with the `California` dataset. It contains 8 attributes of 20,640 districts in California and the goal was to predict the median house value in each district. Here we created a balanced classification task by predicting whether the house value is below or above the median (10,317 positive).

``` python
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# Inspect
df.head()housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Create the binary target variable
median_value = df['MedHouseVal'].median()
df['Above_Median'] = (df['MedHouseVal'] > median_value).astype(int)
df = df.drop('MedHouseVal', axis=1)
df['Above_Median'].value_counts()
```

We do a simple train-test spliting to first fit and evaluate two popular methods: logistic regression and XGBoost.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the data
X = df.drop('Above_Median', axis=1)
y = df['Above_Median']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train the model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_lr = log_reg.predict(X_test_scaled)

# Evaluate the model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistic Regression Accuracy: {accuracy_lr:.4f}')
```
```python
from xgboost import XGBClassifier

# Train the model
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_clf.predict(X_test)

# Evaluate the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f'XGBoost Accuracy: {accuracy_xgb:.4f}')
```

### Classification of Tabular Data with LLM

To let an LLM make prediction on tabular data, we need to transform the rows of tabular data and describe our task into text prompts. This involves:

- Serialize feature names and values into  natural-language string
- Add task-specific prompt

For example, the dataset California contains 8 features named `MedInc`,	`HouseAge`,	`AveRooms`,	`AveBedrms`, `Population`, `AveOccup`,	`Latitude`,	`Longitude`, and one specifc row of the data has the corresponding value (8.33,	41.0,	6.98,	1.02,	322.0,	2.56,	37.88,	-122.23). One way to serialize them is to simply write 
```
- MedInc: 8.33
- HouseAge: 41.0
- AveRooms: 6.98
- AveBedrms: 1.02
- Population: 322.0
- AveOccup: 2.56
- Latitude: 37.88
- Longitude: -122.23
```
and add our question for LLM to anwser：

```
Is this house block valuable (Yes or No)?
Answer:
```
The code for above template would look like this


```python
# Define a template for the prompt as per your new format
def create_prompt(row):
    # Serialization of row data
    serialization = '\n'.join([f'- {col}: {val}' for col, val in row.items()])
    # Combine serialization with the question and answer template
    prompt = f"{serialization}\nIs this house block valuable? Yes or No?\nAnswer:\n|||\n"
    return prompt

# Apply to training and test sets
X_train_prompts = X_train.apply(create_prompt, axis=1)
X_test_prompts = X_test.apply(create_prompt, axis=1)
```

Now we can load the LLM and see how it predict on an example prompt. The logic here is that we let the LLM pick the next word with higher probability between 'Yes' and 'No', and set it as its prediction on this data. 

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Load the tokenizer and model for GPT-2
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Function to make zero-shot predictions based on evaluating the logits
def zero_shot_predict(prompt):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Get the logits from the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the logits for the last token in the sequence
    logits = outputs.logits[:, -1, :]
    
    # Tokens for 'Yes' and 'No'
    yes_token = tokenizer.convert_tokens_to_ids('Yes')
    no_token = tokenizer.convert_tokens_to_ids('No')
    
    # Compare the logits for 'Yes' and 'No'
    yes_logit = logits[:, yes_token].item()
    no_logit = logits[:, no_token].item()
    
    # Choose the token with the higher logit
    if yes_logit > no_logit:
        return 'Yes'
    else:
        return 'No'

# Example prediction
example_prompt = X_test_prompts.iloc[0]
print(zero_shot_predict(example_prompt))
```
:::{admonition} Exercise
:class: tip
Try different formats for the prompt and see if performance changes.
:::

Once we perform empirical studies, it is expected that the predictions from a relatively small model like GPT-2 would approximate random guessing in binary classification tasks. 

How about using fine-tuning? Here we use GPT-2 with a classification head, let the labels correspond to Yes and No, and fine-tune this model on our training examples.

```python
import numpy as np
import evaluate
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, GPT2ForSequenceClassification, DataCollatorWithPadding

# Load the tokenizer and model for GPT-2 adapted for sequence classification
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)

# Set pad token if not already defined (GPT-2 does not have a default pad token)
tokenizer.pad_token = tokenizer.eos_token  
model.config.pad_token_id = model.config.eos_token_id

train_size = 2000
# Update the dataset dictionary
train_data = {'text': X_train_prompts[:train_size], 'labels': y_train[:train_size]}
test_data = {'text': X_test_prompts.tolist(), 'labels': y_test}

train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

# Tokenization function that maps texts to model inputs
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load the accuracy metric
accuracy_metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)
```

Set the hyperparameters and start training:

```python
epochs = 5
train_batch_size=8
test_batch_size=16
lr=1e-5

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=test_batch_size,
    evaluation_strategy='epoch',
    save_strategy="epoch",
    logging_dir='./logs',
    learning_rate=lr,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

# Define a data collator
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize Trainer with compute_metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# Start fine-tuning
trainer.train()
```

:::{admonition} Exercises
:class: tip

- Evaluate the performance of the fine-tuned LLM and compare it against traditional baseline models.
- Reflect on potential "killer applications" where LLM-based approaches could excel over traditional methods.
:::    




## The Curse of Catastrophic Forgetting

### Naive Fine-tuning Leads to Catastrophic Forgetting

Naive fine-tuning involves taking a pre-trained model and training it further on new data without using special mechanisms to preserve previously learned knowledge. This often leads to catastrophic forgetting, where the model loses performance on older tasks while adapting to new ones. This is a critical issue in lifelong learning, where the goal is to continuously learn and retain knowledge over time without requiring retraining on all past data. 

### Catastrophic Forgetting and Lifelong Learning

Lifelong learning is the ability of a model to continuously learn from new experiences without forgetting previously acquired skills. It is important because in many real-world applications such as robotics, personalized AI, data arrives sequentially and the model must adapt without losing its effectiveness on past tasks.

However, catastrophic forgetting presents a major challenge, as traditional training methods tend to overwrite previously learned information.


### Methods in Literature

Several strategies have been developed in the literature to address catastrophic forgetting. A detailed review can be found in this [paper](https://arxiv.org/abs/2409.13997).

- **Regularization**: These methods, such as Elastic Weight Consolidation (EWC), Synaptic Intelligence (SI), and Memory Aware Synapses (MAS), work by identifying important weights from previously learned tasks and penalizing their changes during training on new data. This helps the model retain critical knowledge by constraining updates to crucial weights.

- **Replay**: Experience replay and generative replay are techniques where past data or synthetic samples generated by the model are replayed during training on new tasks. This approach effectively interleaves old and new information, helping maintain performance across tasks.

- **Architectural change**: Methods like Progressive Neural Networks and Dynamically Expandable Networks allocate separate modules or resources for different tasks. By isolating task-specific learning, these architectures prevent interference between tasks and preserve learned information.

- **Distillation**: Knowledge distillation techniques transfer knowledge from an old model (teacher) to a new model (student), allowing the student to retain learned behavior while adapting to new tasks. 

### Example of Catastrophic Forgetting

We demonstrate the catastrophic forgetting issue in LLMs using GPT-2 in a class-incremental setting. 

Specifically, the model is first trained on Task A, learning to classify two sentiment-related classes. Then, without revisiting the data from Task A, the model is fine-tuned on Task B, where it learns to classify new topic-related classes.

- Task A: Sentiment Analysis (IMDb Dataset)
    - This task involves classifying text into two sentiment categories: positive and negative. We use the IMDb dataset, which contains movie reviews labeled as either positive (class 1) or negative (class 0). This task helps the model learn to distinguish sentiment-related content.

- Task B: Topic Classification (AG News Dataset) 
    - In this task, the model classifies text into two topic categories: Technology and Sports. The AG News dataset is used, where we filter the dataset to include only technology (class 2) and sports (class 3) categories. This task introduces new, non-overlapping classes that require the model to learn different types of information.

Here is the sample code to play with.
    
```python
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import torch

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Load and preprocess Task A dataset (IMDb for Sentiment Analysis)
# Remap sentiment labels to class indices: 0 for negative, 1 for positive
train_task_a = load_dataset("imdb", split="train[:2000]")  # Train set subset
test_task_a = load_dataset("imdb", split="test[:500]")  # Test set subset
train_task_a = train_task_a.map(lambda x: {'label': 0 if x['label'] == 0 else 1})  # Remap labels
test_task_a = test_task_a.map(lambda x: {'label': 0 if x['label'] == 0 else 1})
train_task_a = train_task_a.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128), batched=True)
test_task_a = test_task_a.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128), batched=True)
train_task_a.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_task_a.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Load and preprocess Task B dataset (AG News for Topic Classification)
# Remap topic labels to class indices: 2 for Tech, 3 for Sports
train_task_b = load_dataset("ag_news", split="train[:2000]")  # Train set subset
test_task_b = load_dataset("ag_news", split="test[:500]")  # Test set subset
train_task_b = train_task_b.filter(lambda x: x['label'] in [0, 1])  # Filtering for Tech (0) and Sports (1)
test_task_b = test_task_b.filter(lambda x: x['label'] in [0, 1])
train_task_b = train_task_b.map(lambda x: {'label': x['label'] + 2})  # Remap labels to 2 and 3
test_task_b = test_task_b.map(lambda x: {'label': x['label'] + 2})
train_task_b = train_task_b.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128), batched=True)
test_task_b = test_task_b.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128), batched=True)
train_task_b.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_task_b.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Define data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define a simple compute metrics function for accuracy
compute_metrics = lambda eval_pred: {
    'accuracy': (torch.tensor(eval_pred.predictions).argmax(dim=1) == torch.tensor(eval_pred.label_ids)).float().mean().item()
}

# Load pre-trained model and tokenizer with a classification head
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=4)
model.config.pad_token_id = model.config.eos_token_id


# Fine-tuning on Task A (Sentiment Analysis)
training_args_a = TrainingArguments(
    output_dir="./results_task_a",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    eval_strategy="epoch"
)

trainer_a = Trainer(
    model=model,
    args=training_args_a,
    train_dataset=train_task_a,
    eval_dataset=test_task_a,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
print("Training on Task A (Sentiment Analysis)...")
trainer_a.train()

# Evaluate performance on Task A after Task A fine-tuning
results_task_a_after_a = trainer_a.evaluate(test_task_a)
results_task_b_after_a = trainer_a.evaluate(test_task_b)
print("Accuracy on Task A after Task A fine-tuning:", results_task_a_after_a["eval_accuracy"])
print("Accuracy on Task B after Task A fine-tuning:", results_task_b_after_a["eval_accuracy"])

# Fine-tuning on Task B (Topic Classification)
training_args_b = TrainingArguments(
    output_dir="./results_task_b",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch"
)

trainer_b = Trainer(
    model=model,
    args=training_args_b,
    train_dataset=train_task_b,
    eval_dataset=test_task_b,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
print("Training on Task B (Topic Classification)...")
trainer_b.train()

# Evaluate performance on Task A and Task B after Task B fine-tuning
results_task_a_after_b = trainer_b.evaluate(test_task_a)
results_task_b_after_b = trainer_b.evaluate(test_task_b)
print("Accuracy on Task A after Task B fine-tuning:", results_task_a_after_b["eval_accuracy"])
print("Accuracy on Task B after Task B fine-tuning:", results_task_b_after_b["eval_accuracy"])
```



# Reference


- Supervised Fine-tuning Trainer. [code](https://huggingface.co/docs/trl/main/en/sft_trainer)

- Prefix-Tuning: Optimizing Continuous Prompts for Generation. [paper](https://arxiv.org/pdf/2101.00190)

- LoRA: Low-Rank Adaptation of Large Language Models. [paper](https://arxiv.org/pdf/2106.09685)

- ColA: Collaborative Adaptation with Gradient Learning. [paper](https://arxiv.org/pdf/2404.13844)

- List of Large Language Model applied to Tabular Data. [papers](https://github.com/johnnyhwu/Awesome-LLM-Tabular)

- Drift to Remember. [paper](https://arxiv.org/abs/2409.13997)

This lecture note includes code examples and concepts adapted from the following sources. We acknowledge and thank the authors for their contributions to the open-source community.

- Stanford Alpaca: An Instruction-following LLaMA Model. [code](https://github.com/tatsu-lab/stanford_alpaca)

- Self-Instruct: Aligning LM with Self Generated Instructions
. [paper](https://arxiv.org/abs/2212.10560), [code](https://github.com/yizhongw/self-instruct)

- 52K instruction-following data. [data](https://huggingface.co/datasets/yahma/alpaca-cleaned)
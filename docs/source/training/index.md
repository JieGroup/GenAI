
# 3. Training LLMs from Scratch

## Introduction


Building a large language model from scratch has traditionally been the domain of large organizations with extensive computational resources and specialized expertise. However, advancements in technology and an increasing openness in sharing research methodologies have begun to democratize this process.

While university researchers and smaller organizations may still face resource constraints, they now have greater access to tools and techniques that make it feasible to develop custom LLMs. This not only opens up possibilities for advancing research in fields beyond text data, such as AI for science, but also enables a wider range of applications that can drive innovation and contribute to the broader community.


In this chapter, we will outline the process of building your own (albeit smaller) LLM from the ground up, covering essential steps such as architecture design, data curation, effective training, and evaluation techniques.


The first and most crucial step in building an LLM is defining its purpose. This decision influences the model's size, the amount of training data needed, and the computational resources required. Key reasons for creating your own LLM include:

- **Domain-Specificity**: Training with industry-specific data.
- **Greater Data Security**: Incorporating sensitive or proprietary information securely.
- **Ownership and Control**: Retaining control over confidential data and improving the model over time.

The diagram provides an overview of the typical steps involved in training a language model.

```{mermaid}
graph TD
    A[Data Preparation] --> B[Architecture Configuration]
    B --> C[Model Training]
    C --> D[Decoding]
    D --> E[Application Integration]

    style A fill:#f4d03f,stroke:#333,stroke-width:2px
    style B fill:#85c1e9,stroke:#333,stroke-width:2px
    style C fill:#a3e4d7,stroke:#333,stroke-width:2px
    style D fill:#f7dc6f,stroke:#333,stroke-width:2px
    style E fill:#d7bde2,stroke:#333,stroke-width:2px
```



## Transformer Model Architecture


Let us break down the components of the Transformer model, including the embedding layer, transformer blocks, and attention mechanism. 


### Input and Output

The Transformer class initializes the model parameters and sets up the embedding layer.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size

        # Convert input token indices into dense vector representations
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        # Add transformer blocks here
        ...

        # Convert the final hidden state of the model back into a distribution over the vocabulary
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Weight Tying: using the same weight matrix to reduce complexity
        self.tok_embeddings.weight = self.output.weight

        # Precompute positional embeddings
        self.freqs_cos, self.freqs_sin = ...

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h, self.freqs_cos[:seqlen], self.freqs_sin[:seqlen])
        h = self.norm(h)

        if targets is not None: # training-stage
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else: # inference-stage: only select the hidden state of the last token in each sequence
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        return logits
```

where ModelArgs contains the following hyperparameters (with the Llama 7B model default):

```python
from dataclasses import dataclass
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
```

### Transformer Blocks
The model contains a series of TransformerBlock layers, each consisting of an attention mechanism and a feed-forward network.

```python
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        ...
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))
        
        # Normalizes the input to the attention and feed-forward layers.
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        ...

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=args.hidden_dim, multiple_of=args.multiple_of, dropout=args.dropout)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
```

> Unlike **batch normalization**, which normalizes across the batch dimension, **layer normalization** normalizes across the features of each individual token in a sequence. It works independently for each token without relying on the batch statistics. It is applied at every transformer layer to stabilize training by keeping token representations within a consistent range.




### Attention

The attention mechanism allows the model to weigh different parts of the input sequence differently when producing the token embeddings. This is crucial in capturing context-specific relationships between tokens. Attention mechanisms are used in self-attention for transformers or other forms like cross-attention in encoder-decoder models.

The general formula for a multi-head scaled dot-product attention mechanism on input tensors $Q, K, V \in \mathbb{R}^{\ell \times d_m}$ is defined as follows:

$$
\begin{aligned}
O & =\left(H_1 H_2 \cdots H_h\right) W^O \\
H_i & =S_i V_i^{\prime} \\
S_i & =\operatorname{softmax}\left(\frac{Q_i^{\prime} K_i^{\prime \top}}{\sqrt{d_k}}\right) \\
V_i^{\prime} & =V W_i^V \\
K_i^{\prime} & =K W_i^K \\
Q_i^{\prime} & =Q W_i^Q
\end{aligned}
$$

where 
- $O$ is the output
- $H_i$ is the output of the $i$-th attention head
- $S_i$ is the attention score matrix for the $i$-th head
- $Q_i^{\prime}, K_i^{\prime}, V_i^{\prime}$ are the query, key, value projections for the $i$-th head, respectively
- $\ell$ (`seq_len`) is the max sequence length
- $d_m$ (`dim`) is the model dimension
- $h$ (`num_heads`) is the number of attention heads
- $d_k$ and $d_v$ (`head_dim`) are the dimensions of the key and value projections

The projection matrices $W_i^Q, W_i^K \in \mathbb{R}^{d_m \times d_k}$ and $W_i^V \in \mathbb{R}^{d_m \times d_v}$ are learnable parameters. Typically,$d_k=d_v=d_m / h$, meaning the dimensions of the keys and values are set so that the overall dimension is evenly split across the multiple heads.

#### Self-attention

In the self-attention mechanism, $Q, K$, and $V$ are all set to the input $X$.
The process can be summarized as follows:

- Linear Projections: The input $X$ is linearly projected into queries ($Q$), keys ($K$), and values ($V$) using the learned matrices $W_i^Q, W_i^K, W_i^V$.
  - Input $X$ has shape `(batch_size, seq_len, dim)`
  - After linear projections: $Q, K, V$ have shapes `(batch_size, seq_len, num_heads, head_dim)` with `head_dim = dim / num_heads`
  
- Scaled Dot-Product Attention: The dot product of queries and keys is computed, scaled by $\sqrt{d_k}$, and passed through a softmax function to obtain the attention scores.
  - Scores $S = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$ have shape `(batch_size, num_heads, seq_len, seq_len)`
  
- Weighted Sum of Values: The attention scores are used to weigh the values.
  - The weighted sum $\text{output} = S V$ results in a tensor with shape `(batch_size, num_heads, seq_len, head_dim)`
  
- Concatenate and Project: The outputs from all attention heads are concatenated and linearly projected back to the model dimension $d_m$.
  - Final output shape: `(batch_size, seq_len, dim)`
 
#### Self-attention implementation

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # QKV projections
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)

        # Final projection into the residual stream
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
      
        # Create a mask for causal attention
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        self.mask = torch.triu(mask, diagonal=1)
        ...

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
  
        # reorganize dimensions and apply relative positional embeddings to update xq, xk using freqs_cos, freqs_sin
        ...

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores + self.mask[:, :, :seqlen, :seqlen] 
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        output = self.wo(output)
        return output
```

Here, we used Causal Masking `mask` to ensure that the future positions have zero contribution in the weighted sum, which prevents attending to future tokens.

### Model Summary and Example

A complete Pytorch implementation of the model is included in [model.py](https://drive.google.com/file/d/1SU7jSZI36KGwBv5-zgc3WkStPK6lGKwL/view?usp=sharing).

In training our earlier demo model with FileID `15CpwmPuO4p54ZGcJBnyghs_Ns35Ci34O`, we used the following config:
```python
class ModelArgs:
    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: Optional[int] = 6
    vocab_size: int = 2048
    hidden_dim: Optional[int] = None
    multiple_of: int = 32
    norm_eps: float = 1e-5
    max_seq_len: int = 256
    dropout: float = 0.1
```

This corresponds to the architecture:

```{mermaid}
graph LR
    A[Input Embeddings] --> B[Decoder Block 1]
    B --> C[Decoder Block 2]
    C --> D[Decoder Block 3]
    D --> E[Decoder Block 4]
    E --> F[Decoder Block 5]
    F --> G[Decoder Block 6]
    G --> H[Output Layer]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#ccf,stroke:#333,stroke-width:2px
    style C fill:#ccf,stroke:#333,stroke-width:2px
    style D fill:#ccf,stroke:#333,stroke-width:2px
    style E fill:#ccf,stroke:#333,stroke-width:2px
    style F fill:#ccf,stroke:#333,stroke-width:2px
    style G fill:#ccf,stroke:#333,stroke-width:2px
    style H fill:#f9f,stroke:#333,stroke-width:2px
```

and specifically:
```
Transformer(
  (tok_embeddings): Embedding(2048, 288)
  (dropout): Dropout(p=0.1, inplace=False)
  (layers): ModuleList(
    (0-5): 6 x TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=288, out_features=288, bias=False)
        (wk): Linear(in_features=288, out_features=288, bias=False)
        (wv): Linear(in_features=288, out_features=288, bias=False)
        (wo): Linear(in_features=288, out_features=288, bias=False)
        (attn_dropout): Dropout(p=0.0, inplace=False)
        (resid_dropout): Dropout(p=0.0, inplace=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=288, out_features=768, bias=False)
        (w2): Linear(in_features=768, out_features=288, bias=False)
        (w3): Linear(in_features=288, out_features=768, bias=False)
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
  )
  (norm): RMSNorm()
  (output): Linear(in_features=288, out_features=2048, bias=False)
)
```

As seen above, this model consists of:

  - `Embedding(32000, 288)`: Maps input tokens from a vocabulary of 2048 to 288-dimensional embeddings
  - `Dropout(p=0.0)`: Applies dropout with a probability of 0.1
  - **6 x TransformerBlock**: Six layers of Transformer blocks, each containing:
    - **Attention Mechanism**
      - `Linear(in_features=288, out_features=288)`: Four linear transformations (for queries, keys, values, and output) in the multi-head attention mechanism
      - `Dropout(p=0.1)`: Two dropout layers for attention and residual dropout
    - **FeedForward Network**
      - `Linear(in_features=288, out_features=768)`: First linear layer
      - `Linear(in_features=768, out_features=288)`: Second linear layer that projects back to 288 features
      - `Dropout(p=0.1)`: Dropout layer in the feed-forward network
    - **Normalization**
      - `RMSNorm()`: Normalization layer for both the output of the attention block and the feed-forward network
- **Final Normalization**
  - `RMSNorm()`: Normalization layer after the last Transformer block.
- **Output Projection**
  - `Linear(in_features=288, out_features=2048)`: Linear layer that projects the output of the Transformer to a vocabulary size.




## Various Model Architectures

The architecture of the neural network determines the model's capabilities. The transformer architecture becomes a standard for LLMs due to its ability to handle long-range dependencies and parallel processing.

We have covered the decoder-only architecture, as exemplified by the GPT family of models. However, there are other key architectures that use attention mechanisms in different ways:

| Architecture | Description | Suitable for |
|--------------|-------------|--------------|
| **Generative Pre-trained Transformer (GPT)** | Decoder-only: Suited for generative tasks and fine-tuned with labeled data on discriminative tasks. Given the unidirectional architecture, context only flows forward. The GPT framework helps achieve strong natural language understanding using a single-task-agnostic model through generative pre-training and fine-tuning. | Textual entailment, sentence similarity, question answering. |
| **Bi-directional Encoder Representation from Transformers (BERT)** | Encoder-only: Focus on understanding input sequences by applying self-attention over the entire input bidirectionally, making it suitable for tasks like classification. | Classification and sentiment analysis |
| **Text-To-Text Transformer (Sequence-to-Sequence models)** | Encoder-Decoder: Utilize both an encoder to process the input and a decoder to generate the output. With a bidirectional architecture, context flows in both directions. | Translation, Question & Answering, Summarization. |
| **Mixture of Experts (MoE)** | Designed to scale up model capacity by converting dense models into sparse models. The MoE layer consists of many expert models and a sparse gating function. The gates route each input to the top-K best experts during inference. | Generalize well with computational efficiency during inference |
| **Group Query Attention (GQA)** | GQA is an alternative to the multi-head self-attention mechanism. Queries are grouped together based on their similarity or other criteria, leading to shared key and value representations for each query group. | Generalize well with computational and memory efficiency during inference. |


All these architectures leverage the attention mechanism. The main difference lies in how attention is applied:
- Decoder-only models use unidirectional attention to predict the next token in a sequence
- Encoder-only focuses on bidirectional context
- Encoder-decoder uses a decoder that attends to both the input (through encoder-decoder attention) and its own past outputs (through self-attention)


## Data Curation

High-quality, vast amounts of data are essential for training an LLM. The quality of data determines the model's accuracy, bias, predictability, and resource utilization.

A general rule of thumb in language model development is that a larger model has a larger capability. Consequently, a considerable amount of data are often curated to train such models. To better illustrate this relationship between model size and data requirements, here is a comparison of a few existing LLMs:

| Model         | # of Parameters | # of Tokens      |
|---------------|-----------------|------------------|
| GPT-3         | 175 billion     | 0.5 trillion     |
| Llama 2       | 70 billion      | 2 trillion       |
| Falcon 180B   | 180 billion     | 3.5 trillion     |

For better context, consider that 100,000 tokens roughly equate to about 75,000 words, or the length of a typical novel. GPT-3, for example, was trained on hundreds of billions of tokens, which is approximately equivalent to the content of several million novels.

### Characteristics of a High-Quality Dataset
- Filtered for inaccuracies
- Minimal biases and harmful speech
- Cleaned of misspellings, variations, boilerplate text, markup, etc.
- Deduplication
- Privacy redaction
- Diverse in formats and subjects

### Where to Source Data For Training an LLM?

- **Existing Public Datasets**: Data that has been previously used to train LLMs and made available for public use. Prominent examples include:
  - [The Common Crawl](https://www.commoncrawl.org/overview): A dataset containing terabytes of raw web data extracted from billions of pages
  - [The Pile](https://pile.eleuther.ai/): A dataset that contains data from 22 data sources across 5 categories:
	  - *Academic Writing*: e.g., arXiv
	  - *Online or Scraped Resources*: e.g., Wikipedia
	  - *Prose*: e.g., Project Gutenberg
	  - *Dialog*: e.g., YouTube subtitles
	  - *Miscellaneous*: e.g., GitHub
  - [StarCoder](https://arxiv.org/pdf/2305.06161): Near 800GB of coding samples in several programming languages
  - [Hugging Face](https://huggingface.co/datasets): An online community with over 100,000 public datasets

- **Private Datasets**: Curated in-house or purchased
- **Directly From the Internet**: Less recommended due to potential inaccuracies/biases


### Example Dataset: TinyStories 

The [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) is a collection of short, simple stories designed to train and evaluate language models on narrative understanding and generation. 

As before, use [utils.py](https://drive.google.com/file/d/1tKQCXmrT4whJr1V33nBVRhaNzniRT5KE/view?usp=sharing) and run the following to download the TinyStories dataset. A bunch of json files will be created within `TinyStories_all_data` under your specified directory `data_dir`.

```python
from utils import download_TinyStories
download_TinyStories(data_dir="demo_data")
```

### Tokenization

We went through [tokenization and vocabulary](https://genai-course.jding.org/en/latest/llm/index.html#tokenization-and-vocabulary) and discussed their tradeoffs. Here is an implementation of it:

```python
import glob
import json
import os
from tqdm import tqdm
import sentencepiece as spm

def train_vocab(vocab_size, data_dir, dataset_name="TinyStories_all_data"):
    """
    Trains a custom sentencepiece tokenizer on the TinyStories dataset.
    It produces a file saved in "tok{vocab_size}" under the data_dir directory.
    """

    assert vocab_size > 0, "Vocab size must be positive"
    prefix = os.path.join(data_dir, f"tok{vocab_size}") #output file prefix
    
    # Export a number of shards into a text file for vocab training. The lower the more efficiency
    num_shards = 10
    temp_file = os.path.join(data_dir, "temp.txt")
    data_dir = os.path.join(data_dir, dataset_name)
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    print(f"Writing temporary file {temp_file} with {num_shards} shards...")
    with open(temp_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                text = example["story"]
                text = text.strip()
                of.write(text + "\n")
    print(f"Size: {os.path.getsize(temp_file) / 1024 / 1024:.2f} MB")
    print("Train the sentencepiece model ...")
    spm.SentencePieceTrainer.train(input=temp_file, model_prefix=prefix, model_type="bpe", vocab_size=vocab_size, split_digits=False, allow_whitespace_only_pieces=True, byte_fallback=True, unk_surface=r" \342\201\207 ", normalization_rule_name="identity")
    os.remove(temp_file)
    tokenizer_model = f"{prefix}.model"
    print(f"Trained tokenizer is in {tokenizer_model}")
    return tokenizer_model

tokenizer_model = train_vocab(2048, "demo_data", "TinyStories_all_data")
```

Using the above generated tokenizer model, we can pretokenizer the original text data for model training. Use the Python module [tokenizer.py](https://drive.google.com/file/d/1uXCgdmip79J6efM5hiHGCy9mdr_U8BXT/view?usp=sharing).

```python
from tokenizer import pretokenize
output_bin_dir = pretokenize(data_dir="demo_data", dataset_name="TinyStories_all_data", tokenizer_model=tokenizer_model)
# This will create a directory called TinyStories_all_data_pretok under data_dir
```

### Putting the Model and Data together

#### Configuration parameters

Set up the configuration (including hyper-) parameters in a [demo_training_config.py](https://drive.google.com/file/d/1bxSYERj2F81n5WkzLukOj355Zhccf-JG/view?usp=sharing) module, and then load it for training, e.g.,

```python
from demo_training_config import Config
config = Config()
config_dict = config.to_dict()

# Load configuration parameters
pretok_bin_dir = config.pretok_bin_dir
model_out_dir = config.model_out_dir
...
```

#### Training

A full training script can be downloaded at [demo_train.py](https://drive.google.com/file/d/1Jr1eQvJCo9MKlWryCpGnd8s0gRsFvH6v/view?usp=sharing). This script sets up and trains a model from scratch, based on the earlier defined Transformer architecture, downloaded data, and hyperparameters.  

The following is a simplified sketch of the training procedure.

```python
from model import Transformer, ModelArgs
from tokenizer import BatchProcessor
import math
import os
import time
from contextlib import nullcontext
from functools import partial
import torch

# Set up mixed precision
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# Create batches using partial function
iter_batches = partial(BatchProcessor.iter_batches, batch_size=config.batch_size, device=config.device, ...)

# Initialize model and optimizer
model_args = ModelArgs(dim=config.dim, n_layers=config.n_layers, n_heads=config.n_heads, ...)
model = Transformer(model_args)
model.to(config.device)
optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, ...)

# Function to estimate loss on validation data
@torch.no_grad()
def estimate_loss():
    model.eval()
    # Evaluate loss...
    model.train()
    return train_loss, val_loss

# Training loop
train_batch_iter = iter_batches(split="train")
while iter_num <= config.max_iters:
    # Adjust learning rate and evaluate periodically
    if iter_num % config.eval_interval == 0:
        losses = estimate_loss()
        # Save checkpoints ...

    # Forward and backward pass with optional gradient accumulation
    for micro_step in range(config.gradient_accumulation_steps):
        with ctx:
            logits = model(X, Y)
            loss = model.last_loss / gradient_accumulation_steps
        # Fetch next batch and backpropagate
        X, Y = next(train_batch_iter)
        scaler.scale(loss).backward()

    # Optimizer step and update
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

> In the above, `set_to_none=True` saves memory by not allocating memory for zeroed-out gradients until the next backward pass computes new gradients.

> `ctx` is a context manager for automatic mixed precision training. Operations within the `with ctx` block are performed in mixed precision to improve training speed and reduce memory usage on compatible GPUs. So certain computations are automatically done in 16-bit (half precision) instead of the standard 32-bit (full precision). 

> While ctx manages mixed precision during the forward pass, GradScaler handles the scaling of the loss during the backward pass in mixed precision training. It aims to prevent gradient underflow by scaling up the loss value before backpropagation.


:::{admonition} Exercise 1
:class: tip

[BertViz](https://github.com/jessevig/bertviz) is an interactive tool for visualizing attention in language models such as BERT, GPT-2, or T5. Use this tool to explore the attention mechanism by visualizing the attention patterns of a few selected layers and heads on different input examples. This exercise will be useful for your upcoming homework3.

Consider questions such as:
- How does attention evolve throughout the model's layers?
- How does attention evolve throughout the model training?
- Are there noticeable patterns in attention when comparing simple versus complex sentences?
:::


:::{admonition} Exercise 2
:class: tip

A crucial question in any learning task is: **How do we assess whether further model improvement is necessary at the current training stage?**

In classical settings such as regression and classification with small output spaces, there exist statistical diagnostic tools to quantify whether a model can be improved further or has reached its theoretical performance limit (given the data). For instance, [this paper](https://par.nsf.gov/servlets/purl/10347493#:~:text=To%20our%20best%20knowledge%2C%20there,challenging%20to%20construct%20proper%20tests) formulates classifier diagnostics from a hypothesis testing perspective.

**Discuss** potential approaches (both practical and theoretical) to decide if we should invest more resources in continuing to train a model or update its architecture. Consider aspects beyond validation performance as this alone does not address the problem. 
:::

:::{admonition} Exercise 3
:class: tip

**How Long Does It Take to Train an LLM From Scratch?**

The training time varies significantly depending on several factors, such as model complexity, training dataset, computational resources, choices in hyperparameters, and task evaluation criteria.

**Discuss** the equation(s) or methods that could be used to predict the total training time based on an initial pilot run, and clearly define what constitutes a pilot run in this context.
:::



## Evaluating Your Bespoke LLM

Evaluating a large language model is inherently complex and often subjective, as different use cases and societal considerations can influence what we deem as "successful" or "safe" model behavior. For instance, an LLM's performance may be measured not only by its ability to understand and generate language accurately but also by how it adheres to ethical and safety standards, which can vary significantly across cultures and regulatory regimes. Thus, evaluation is not just a technical exercise but also a reflection of diverse priorities and values. We will revisit this in later chapters on human value alignment and AI safety.

Despite these complexities, standardized benchmarks offer a valuable starting point for assessing an LLM's general capabilities. These benchmarks provide objective measures of the model's language understanding, reasoning, and problem-solving skills, serving as an initial evaluation in a zero-shot setting. By using these benchmarks, we can establish a baseline for how well the model performs before considering further task-specific fine-tuning or adjustments.

Some of the most widely used benchmarks for evaluating LLM performance include:

- **[ARC](https://arxiv.org/abs/1803.05457)**: The AI2 Reasoning Challenge is a question-answering (QA) benchmark designed to evaluate the model's knowledge and reasoning skills, particularly focusing on scientific and commonsense reasoning.

- **[HellaSwag](https://arxiv.org/abs/1905.07830)**: This benchmark uses sentence completion exercises to test commonsense reasoning and natural language inference (NLI) capabilities, challenging models to predict the most plausible continuation of a given scenario.

- **[MMLU](https://arxiv.org/abs/2009.03300)**: The Massive Multitask Language Understanding benchmark consists of 15,908 questions across 57 tasks, measuring natural language understanding (NLU) by testing the model's ability to comprehend and solve diverse problems across various domains.

- **[TruthfulQA](https://arxiv.org/abs/2109.07958)**: This benchmark measures a model’s ability to generate truthful answers, evaluating its propensity to "hallucinate" or generate misleading information, which is crucial for ensuring the reliability of generated content.

- **[GSM8K](https://arxiv.org/abs/2110.14168)**: This dataset assesses multi-step mathematical reasoning abilities through a collection of 8,500 grade-school-level math word problems, challenging models to handle complex arithmetic and logical reasoning.

- **[HumanEval](https://arxiv.org/abs/2107.03374)**: This benchmark evaluates an LLM’s ability to generate functionally correct code by measuring how well the model can complete coding tasks that require an understanding of programming logic and syntax.

- **[MT Bench](https://arxiv.org/pdf/2402.14762)**: The Multi-Turn Bench evaluates a language model’s ability to engage effectively in multi-turn dialogues, like those performed by chatbots, assessing coherence and relevance over extended interactions.



## References

- Llama 2: Open foundation and fine-tuned chat models. [paper](https://arxiv.org/pdf/2307.09288)

- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. [paper](https://arxiv.org/pdf/1810.04805)

- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. [paper](https://arxiv.org/pdf/1910.10683)

- Scaling vision with sparse mixture of experts. [paper](https://arxiv.org/pdf/2106.05974)

- GPT-4 Technical Report. [paper](https://arxiv.org/pdf/2303.08774)

- Attention in transformers, visually explained. [video](https://www.youtube.com/watch?v=eMlx5fFNoYc)

- Code for The Annotated Transformer [blog](http://nlp.seas.harvard.edu/annotated-transformer/), [code](https://github.com/harvardnlp/annotated-transformer/)

- Language Models are Unsupervised Multitask Learners (GPT-2). [paper](https://hayate-lab.com/wp-content/uploads/2023/05/61b1321d512410607235e9a7457a715c.pdf), [code](https://github.com/openai/gpt-2)

- A PyTorch re-implementation of [GPT-2](https://github.com/openai/gpt-2), both training and inference. [code](https://github.com/karpathy/nanoGPT)

- The Illustrated Transformer. [blog](https://jalammar.github.io/illustrated-transformer/)

- Is a Classification Procedure Good Enough?—A Goodness-of-Fit Assessment Tool for Classification Learning. [paper](https://par.nsf.gov/servlets/purl/10347493#:~:text=To%20our%20best%20knowledge%2C%20there,challenging%20to%20construct%20proper%20tests)

This lecture note includes code examples and concepts adapted from the following sources. We acknowledge and thank the authors for their contributions to the open-source community.

- llama.c open-source project. [code](https://github.com/karpathy/llama2.c)

- TinyStories: How Small Can Language Models Be and Still Speak
Coherent English? [paper](https://arxiv.org/pdf/2305.07759), [data](https://huggingface.co/datasets/roneneldan/TinyStories)


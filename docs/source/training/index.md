
# 3. Training LLMs from Scratch

## Introduction

Building a large language model (LLM) from scratch was once a task reserved for larger organizations due to the considerable computational resources and specialized knowledge required. Today, with the growing availability of resources and knowledge, organizations of all sizes can develop custom LLMs to enhance productivity, efficiency, and competitive edge.

This chapter details the process of building your own LLM from the ground up, from architecture definition and data curation to effective training and evaluation techniques.



The first and most crucial step in building an LLM is defining its purpose. This influences the model's size, the amount of training data needed, and the computational resources required.

Key reasons for creating your own LLM include:
- **Domain-Specificity**: Training with industry-specific data.
- **Greater Data Security**: Incorporating sensitive or proprietary information securely.
- **Ownership and Control**: Retaining control over confidential data and improving the LLM over time.


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

> Unlike **batch normalization**, which normalizes across the batch dimension, **layer normalization** normalizes across the features of each individual token in a sequence. It works independently for each token without relying on the batch statistics. It is applied at every transformer layer to stabilize training by keeping activations within a consistent range.




### Attention

The attention mechanism allows the model to weigh different parts of the input sequence differently when producing the token embeddings. Attention mechanisms are used in self-attention for transformers or other forms like cross-attention in encoder-decoder models.

- Input $x$ is projected into queries $Q$, keys $K$, and values $V$

  - x has shape `(batch_size, seq_len, dim)`
  - After linear projections: $xq, xk, xv$ have shapes `(batch_size, seq_len, num_heads, head_dim)` with `head_dim = dim / num_heads`
  - Dot product of queries and keys: $s = xq \cdot xk^T$ gives shape `(batch_size, num_heads, seq_len, seq_len)`
  - Weighted sum of values: $\textrm{output}=\textrm{softmax}(s) \cdot xv$ gives shape `(batch_size, num_heads, seq_len, head_dim)`
  - Final Projection: Concatenate across heads and project back to `(batch_size, seq_len, dim)`

- Use Position Embeddings to encode positional information

- Compute scaled dot-product attention: 

$$
\textrm{Attention}(Q,K,V) = \textrm{softmax}(QK^T / \sqrt{d_k}) V
$$

- The attended representation is then projected back to the model's dimension

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
            super().__init__()
            ...
            
    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        # QKV projections
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)

        # Final projection into the residual stream
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        # reorganize dimensions and apply relative positional embeddings to update xq, xk using freqs_cos, freqs_sin
        ...

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        output = self.wo(output)
        return output
```

### Summary and Example

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


All these architectures leverage the attention mechanism. The main difference lies in how attention is applied:
- Decoder-only models use unidirectional attention to predict the next token in a sequence
- Encoder-only focuses on bidirectional context
- Encoder-decoder uses a decoder that attends to both the input (through encoder-decoder attention) and its own past outputs (through self-attention)



### Assembling the Encoder and Decoder

#### Encoder
Converts input sequences into weighted embeddings.
Structure:
- Embedding layer
- Positional encoder
- Self-attention mechanism
- Feed-Forward network
- Normalization layers and residual connections

#### Decoder
Generates output tokens from weighted embeddings produced by the encoder.
  
The decoder has a similar architecture to the encoder, with a couple of key differences:

-   It has two self-attention layers, while the encoder has one.
-   It employs two types of self-attention
    -   **Masked Multi-Head Attention:**  uses a causal masking mechanism to prevent comparisons against future tokens.
    -   **Encoder-Decoder Multi-Head Attention:** each output token calculates attention scores against all input tokens, better establishing the relationship between the input and output for greater accuracy. This cross-attention mechanism also employs casual masking to avoid influence from future output tokens.
    
Structure:
- Embedding layer
- Positional encoder
- Masked self-attention mechanism
- Normalization layer
    -  **Residual connection that feeds into normalization layer**
- Encoder-Decoder self-attention mechanism
	-  **Residual connection that feeds into normalization layer**
- Feed-Forward network
- Normalization layers and residual connections

#### Complete Transformer
Combines multiple encoders and decoders stacked in equal sizes to enhance performance by capturing different characteristics and underlying patterns from the input.


## Data Curation

High-quality, vast amounts of data are essential for training an LLM. The quality of data determines the model's accuracy, bias, predictability, and resource utilization.

A general rule of thumb in language model development is that the more performant and capable you want your Large Language Model (LLM) to be, the more parameters it requires. Consequently, a larger amount of data must also be curated to train such models effectively. To better illustrate this relationship between model size, performance, and data requirements, here's a comparison of a few existing LLMs and the amount of data, in tokens, used to train them:

| Model         | # of Parameters | # of Tokens      |
|---------------|-----------------|------------------|
| GPT-3         | 175 billion     | 0.5 trillion     |
| Llama 2       | 70 billion      | 2 trillion       |
| Falcon 180B   | 180 billion     | 3.5 trillion     |

For better context, consider that 100,000 tokens equate to approximately 75,000 words, or about the length of an entire novel. Therefore, GPT-3, for example, was trained on the equivalent of about 5 million novels' worth of data.

### Characteristics of a High-Quality Dataset
- Filtered for inaccuracies
- Minimal biases and harmful speech
- Cleaned of misspellings, variations, boilerplate text, markup, etc.
- Deduplication
- Privacy redaction
- Diverse in formats and subjects

### Where Can You Source Data For Training an LLM?

There are several places to source training data for your language model. Depending on the amount of data you need, it is likely that you will draw from each of the sources outlined below.

- **Existing Public Datasets**: Data that has been previously used to train LLMs and made available for public use. Prominent examples include:
  - **The Common Crawl**: A dataset containing terabytes of raw web data extracted from billions of pages. It also has widely-used variations or subsets, including RefinedWeb and C4 (Colossal Cleaned Crawled Corpus).
  - **The Pile**: A popular text corpus that contains data from 22 data sources across 5 categories:
	  - **Academic Writing**: e.g., arXiv
	  - **Online or Scraped Resources**: e.g., Wikipedia
	  - **Prose**: e.g., Project Gutenberg
	  - **Dialog**: e.g., YouTube subtitles
	  - **Miscellaneous**: e.g., GitHub
  - **StarCoder**: Close to 800GB of coding samples in a variety of programming languages.
  - **Hugging Face**: An online resource hub and community that features over 100,000 public datasets.

- **Private Datasets**: Curated in-house or purchased.
- **Directly From the Internet**: Less recommended due to potential inaccuracies and biases.


### Example Dataset: TinyStories 

As before, fetch [utils.py](https://drive.google.com/file/d/1tKQCXmrT4whJr1V33nBVRhaNzniRT5KE/view?usp=sharing) and run the following to download the TinyStories dataset. A bunch of json files will be created within `TinyStories_all_data` under your specified directory `data_dir`.

```python
from utils import download_TinyStories
download_TinyStories(data_dir="demo_data")
```

## Tokenization

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

## Putting the Model and Data together


Set up the configuration (including hyper-) parameters in a [demo_training_config.py](https://drive.google.com/file/d/1bxSYERj2F81n5WkzLukOj355Zhccf-JG/view?usp=sharing) module, and then load it for training, e.g.,

```python
from demo_training_config import Config
config = Config()
config_dict = config.to_dict()
pretok_bin_dir = config.pretok_bin_dir
model_out_dir = config.model_out_dir
...
```

A full training script can be downloaded at [demo_train.py](https://drive.google.com/file/d/1Jr1eQvJCo9MKlWryCpGnd8s0gRsFvH6v/view?usp=sharing). 


### LLM Hyperparameters

- **Batch Size**: batch is a collection of instances from the training data, which are fed into the model at a particular timestep. Larger batches require more memory but also accelerate the training process as you get through more data at each interval. Conversely, smaller batches use less memory but prolong training. Generally, it is best to go with the largest data batch your hardware will allow while remaining stable, but finding this optimal batch size requires experimentation.
- **Learning Rate**: how quickly the LLM updates itself in response to its loss function, i.e., its frequency of incorrect prediction, during training. A higher learning rate expedites training but could cause instability and overfitting. A lower learning rate, in contrast, is more stable and improves generalization – but lengthens the training process.
- **Temperature**: adjusts the range of possible output to determine how “creative” the LLM is. Represented by a value between 0.0 (minimum) and 2.0 (maximum), a lower temperature will generate more predictable output, while a higher value increases the randomness and creativity of responses.





### How Long Does It Take to Train an LLM From Scratch?

The training time for an LLM varies based on several factors:
- **Complexity of the Use Case**: More complex tasks require more extensive training.
- **Training Data**: The amount, complexity, and quality of the data significantly impact training time.
- **Computational Resources**: The available hardware affects how quickly the model can be trained.

Training an LLM for simple tasks with small datasets might take a few hours, while more complex tasks with large datasets could take months. Challenges in Training include
- **Underfitting**: Occurs when the model is trained for too short a time and fails to capture relationships in the data.
- **Overfitting**: Happens when the model is trained for too long and learns the training data too well, failing to generalize to new data.

To mitigate these issues, monitor the model's performance and stop training when it consistently produces the expected outcomes and makes accurate predictions on unseen data.



## Conclusion

Building an LLM from scratch involves:
- Defining the use case
- Creating the model architecture
- Data curation
- Training and fine-tuning
- Evaluation

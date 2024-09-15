
# 3. Training LLMs from Scratch

## Introduction

Building a large language model (LLM) from scratch was once a task reserved for larger organizations due to the considerable computational resources and specialized knowledge required. Today, with the growing availability of resources and knowledge, organizations of all sizes can develop custom LLMs to enhance productivity, efficiency, and competitive edge.

This chapter details the process of building your own LLM from the ground up, from architecture definition and data curation to effective training and evaluation techniques.


## Determine the Use Case For Your LLM

The first and most crucial step in building an LLM is defining its purpose. This influences the model's size, the amount of training data needed, and the computational resources required.

Key reasons for creating your own LLM include:
- **Domain-Specificity**: Training with industry-specific data.
- **Greater Data Security**: Incorporating sensitive or proprietary information securely.
- **Ownership and Control**: Retaining control over confidential data and improving the LLM over time.


## Transformer Model Architecture


## Transformer Structure


### Mermaid Diagram for Large Language Model Training Procedure

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


### Transformer Model Architecture

We will look into it through a toy model, which can be downloaded from [here]().  Its architecture is:

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

```
Transformer(
  (tok_embeddings): Embedding(32000, 288)
  (dropout): Dropout(p=0.0, inplace=False)
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
  (output): Linear(in_features=288, out_features=32000, bias=False)
)
```

As seen above, it consists of:

  - `Embedding(32000, 288)`: Maps input tokens from a vocabulary of 32,000 to 288-dimensional embeddings.
  - `Dropout(p=0.0)`: Applies dropout with a probability of 0.0 (effectively no dropout in this setup).
  - **6 x TransformerBlock**: Six layers of Transformer blocks, each containing:
    - **Attention Mechanism**
      - `Linear(in_features=288, out_features=288)`: Four linear transformations (for queries, keys, values, and output) in the multi-head attention mechanism, all with 288 features.
      - `Dropout(p=0.0)`: Two dropout layers for attention and residual dropout, both set to 0.0 probability.
    - **FeedForward Network**
      - `Linear(in_features=288, out_features=768)`: First linear layer of the feed-forward network.
      - `Linear(in_features=768, out_features=288)`: Second linear layer that projects back to 288 features.
      - `Dropout(p=0.0)`: Dropout layer in the feed-forward network.
    - **Normalization**
      - `RMSNorm()`: Normalization layer for both the output of the attention block and the feed-forward network.
- **Final Normalization**
  - `RMSNorm()`: Normalization layer after the last Transformer block.
- **Output Projection**
  - `Linear(in_features=288, out_features=32000)`: Linear layer that projects the output of the Transformer to a vocabulary size of 32,000.




## Create Your Model Architecture

The architecture of the neural network determines the model's capabilities. The transformer architecture is the best choice for LLMs due to its ability to handle long-range dependencies, parallel processing, and capturing underlying patterns from data.

Frameworks like PyTorch and TensorFlow provide the necessary components for neural network development.

Some popular architectures include the following 

| Architecture | Description | Suitable for |
|--------------|-------------|--------------|
| **Bi-directional Encoder Representation from Transformers (BERT)** | Encoder-only architecture, best suited for tasks that can understand language. | Classification and sentiment analysis |
| **Generative Pre-trained Transformer (GPT)** | Decoder-only architecture suited for generative tasks and fine-tuned with labeled data on discriminative tasks. Given the unidirectional architecture, context only flows forward. The GPT framework helps achieve strong natural language understanding using a single-task-agnostic model through generative pre-training and discriminative fine-tuning. | Textual entailment, sentence similarity, question answering. |
| **Text-To-Text Transformer (Sequence-to-Sequence models)** | Encoder-decoder architecture. It leverages the transfer learning approach to convert every text-based language problem into a text-to-text format, that is taking text as input and producing the next text as output. With a bidirectional architecture, context flows in both directions. | Translation, Question & Answering, Summarization. |
| **Mixture of Experts (MoE)** | Model architecture decisions that can be applied to any of the‌ architectures. Designed to scale up model capacity substantially while adding minimal computation overhead, converting dense models into sparse models. The MoE layer consists of many expert models and a sparse gating function. The gates route each input to the top-K (K>=2 or K=1) best experts during inference. | Generalize well across tasks for computational efficiency during inference, with low latency |



### Creating The Transformer’s Components

#### 1. Embedding Layer
Converts input into vector representations. This involves:
- Tokenizing the input.
- Assigning integer IDs to tokens.
- Converting integers into multi-dimensional vectors (embeddings).

#### Positional Encoder
Generates positional encodings added to each embedding to maintain the position of tokens within a sequence.

#### Self-Attention Mechanism
The most crucial component, responsible for comparing each embedding to determine similarity and semantic relevance. Multi-head attention allows parallel processing, enhancing performance and reliability.

#### Feed-Forward Network
Captures higher-level features of the input sequence with sub-layers: 
- First Linear Layer
- Non-Linear Activation Function (e.g., ReLU)
- Second Linear Layer

#### Normalization Layers
Ensures input embeddings fall within a reasonable range, stabilizing the model.
In particular, the transformer architecture utilizes layer normalization, which normalizes the output for each token at every layer – as opposed to batch normalization, for example, which normalizes across each portion of data used during a time step. Layer normalization is ideal for transformers because it maintains the relationships between the aspects of each token; and does not interfere with the self-attention mechanism.

#### Residual Connections
Feeds the output of one layer directly into the input of another, preventing information loss and aiding in faster, more effective training.
During forward propagation, i,e., as training data is fed into the model, residual connections provide an additional pathway that ensures that the original data is preserved and can bypass transformations at that layer. Conversely, during backward propagation, i,e., when the model adjusts its parameters according to its loss function, residual connections help gradients flow more easily through the network, helping to mitigate vanishing gradients, where gradients become increasingly smaller as they pass through more layers.

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

### How Long Does It Take to Train an LLM From Scratch?

The training time for an LLM varies based on several factors:
- **Complexity of the Use Case**: More complex tasks require more extensive training.
- **Training Data**: The amount, complexity, and quality of the data significantly impact training time.
- **Computational Resources**: The available hardware affects how quickly the model can be trained.

Training an LLM for simple tasks with small datasets might take a few hours, while more complex tasks with large datasets could take months. Challenges in Training include
- **Underfitting**: Occurs when the model is trained for too short a time and fails to capture relationships in the data.
- **Overfitting**: Happens when the model is trained for too long and learns the training data too well, failing to generalize to new data.

To mitigate these issues, monitor the model's performance and stop training when it consistently produces the expected outcomes and makes accurate predictions on unseen data.


## Training Your Custom LLM

Training involves forward and backward propagation over multiple batches of data and epochs until the model's parameters converge.

### LLM Training Techniques

#### Parallelization
Distributes training tasks across multiple GPUs.
- **Data Parallelization**: Divides training data into shards.
- **Tensor Parallelization**: Divides matrix multiplications.
- **Pipeline Parallelization**: Distributes transformer layers.
- **Model Parallelization**: Distributes the model across GPUs.

#### Gradient Checkpointing
Gradient checkpointing is a technique used to reduce the memory requirements of training LLMs. It is a valuable training technique because it makes it more feasible to train LLMs on devices with restricted memory capacity. Subsequently, by mitigating out-of-memory errors, gradient checkpointing helps make the training process more stable and reliable.

Typically, during forward propagation, the model’s neural network produces a series of intermediate activations: output values derived from the training data that the network later uses to refine its loss function. With gradient checkpointing, though all intermediate activations are calculated, only a subset of them are stored in memory at defined checkpoints.

During backward propagation, the intermediate activations that were not stored are recalculated. However, instead of recalculating all the activations, only the subset – stored at the checkpoint – needs to be recalculated. Although gradient checkpointing reduces memory requirements, the tradeoff is that it increases processing overhead; the more checkpoints used, the greater the overhead.

#### LLM Hyperparameters
- **Batch Size**: batch is a collection of instances from the training data, which are fed into the model at a particular timestep. Larger batches require more memory but also accelerate the training process as you get through more data at each interval. Conversely, smaller batches use less memory but prolong training. Generally, it is best to go with the largest data batch your hardware will allow while remaining stable, but finding this optimal batch size requires experimentation.
- **Learning Rate**: how quickly the LLM updates itself in response to its loss function, i.e., its frequency of incorrect prediction, during training. A higher learning rate expedites training but could cause instability and overfitting. A lower learning rate, in contrast, is more stable and improves generalization – but lengthens the training process.
- **Temperature**: adjusts the range of possible output to determine how “creative” the LLM is. Represented by a value between 0.0 (minimum) and 2.0 (maximum), a lower temperature will generate more predictable output, while a higher value increases the randomness and creativity of responses.

## Fine-Tuning Your LLM


After training your LLM from scratch with larger, general-purpose datasets, you will have a base, or pre-trained, language model. To prepare your LLM for your chosen use case, you likely have to fine-tune it. Fine-tuning is the process of further training a base LLM with a smaller, task or domain-specific dataset to enhance its performance on a particular use case. Fine-tuning methods broadly fall into two categories: full fine-tuning and transfer learning: 
- **Full Fine-Tuning**: Updates all base model parameters.  This is the most comprehensive way to train an LLM for a specific task or domain – but requires more time and resources.
- **Transfer Learning**: Freezes most layers and tunes specific ones. The remaining layers – or, often, newly added – unfrozen layers are fine-tuned with the smaller fine-tuning dataset – requiring less time and computational resources than full fine-tuning.

## Evaluating Your Bespoke LLM

Evaluation ensures the LLM performs as expected using unseen datasets to avoid overfitting.

### LLM Benchmarks
Standardized tests to objectively evaluate performance. Some of the most widely used benchmarks for evaluating LLM performance include:
- **ARC**: a question-answer (QA) benchmark designed to evaluate knowledge and reasoning skills.
- **HellaSwag**: uses sentence completion exercises to test commonsense reasoning  and natural language inference (NLI) capabilities.
- **MMLU**: a  benchmark comprised of 15,908 questions across 57 tasks that measure natural language understanding (NLU), i.e., how well an LLM _understands_ language and, subsequently, can solve problems.
- **TruthfulQA**: measuring a model’s ability to generate truthful answers, i.e., its propensity to “hallucinate”.
- **GSM8K**: measures multi-step mathematical abilities through a collection of 8,500 grade-school-level math word problems.
- **HumanEval**: measures an LLM’s ability to generate functionally correct code.
- **MT Bench**: evaluates a language model’s ability to effectively engage in multi-turn dialogues – like those engaged in by chatbots.

## Conclusion

Building an LLM from scratch involves:
- Defining the use case
- Creating the model architecture
- Data curation
- Training and fine-tuning
- Evaluation

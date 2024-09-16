# 7. Efficient Deployment Strategies


Large Language Models (LLMs) such as GPT-3/4, Falcon, and Llama are rapidly advancing in their ability to tackle human-centric tasks. Deploying these models in real-world tasks remains challenging due to the computational and memory demands for inference, especially with extensive contextual information. This note is based on [this blog](https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization).


## Model Compiling

Add the following after saving the checkpoint during the training
```python
from export import model_export

# The function model_export serializes the model’s state dictionary to a binary file (model.bin). 
# This file is meant to be a lightweight alternative to the full checkpoint, 
# for deployment or inference where only the model parameters are needed, not the full training state.
model_export(model, os.path.join(out_dir, "model.bin"), version=0)
```

## Lower Precision

Memory requirements for LLMs can be best understood by viewing the LLM as a set of weight matrices and vectors. Typically, LLMs consist of billions of parameters stored in `float32`, `bfloat16`, or `float16` formats.  Model quantization techniques generally trade improved memory efficiency against accuracy and in some cases inference time.

### Memory Requirements

Nowadays, models are however rarely trained in full float32 precision, but usually in bfloat16 precision or less frequently in float16 precision. Therefore the rule of thumb becomes:

For shorter text inputs (less than 1024 tokens), the memory requirement for inference is very much dominated by the memory requirement to load the weights. Therefore, for now, let’s assume that the memory requirement for inference is equal to the memory requirement to load the model into the GPU VRAM.

| Number of Parameters (in billions) | Precision   | VRAM Requirement        |
|------------------------------------|-------------|-------------------------|
| `p`                                | float32     | `4p GB`              |
| `p`                                 | bfloat16    | `2p GB`              |
| `p`                                | float16     | `2p GB`              |


#### Examples

| Model              | Model Size (GB) | VRAM Required (GB) |
|--------------------|------------------|---------------------|
| GPT-3              | 175              | 350                 |
| Bloom              | 176              | 352                 |
| Llama-2-70b        | 70               | 140                 |
| Falcon-40b         | 40               | 80                  |
| MPT-30b            | 30               | 60                  |
| bigcode/starcoder  | 15.5             | 31                  |


Most models are trained in bfloat16 nowadays, so Float32 won’t give better inference results than the precision that was used to train the model. In that case, there is no need to run the model in full float32 precision.

### Model loading

Naive pipeline parallelism is supported out of the box. For this, simply load the model with `device="auto"` which will automatically place the different layers on the available GPUs as explained [here](https://huggingface.co/docs/accelerate/v0.22.0/en/concept_guides/big_model_inference). We will use [bigcode/octocoder](https://huggingface.co/bigcode/octocoder) as it can be run on a single 40 GB A100 GPU device chip

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model = AutoModelForCausalLM.from_pretrained("bigcode/octocoder", torch_dtype=torch.bfloat16, device_map="auto", pad_token_id=0)
tokenizer = AutoTokenizer.from_pretrained("bigcode/octocoder")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"

result = pipe(prompt, max_new_tokens=60)[0]["generated_text"][len(prompt):]
result
print(f"peak GPU memory allocation: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024}")
```

### Quantization Schemes

Quantization schemes are designed to reduce the precision of weights in neural network models, with the goal of maintaining the accuracy of the model's inference results as closely as possible to the original using bfloat16 precision. It was found that quantizing model weights to 8-bit or 4-bit can achieve computational advantages without significant loss in performance.

While there are many quantization approaches, the general process involves:
1. **Quantize Weights**: Convert all model weights to the target precision.
2. **Load Quantized Weights**: Use these weights, ensuring inputs are passed in bfloat16 precision.
3. **Dynamic Dequantization**: During computation, weights are temporarily converted back to bfloat16 from the quantized format to match the precision of input vectors.

Quantization does not necessarily reduce inference time; it may actually increase due to the overhead of dynamic dequantization. To implement weight quantization in Transformers, we use the `bitsandbytes` library
```bash
!pip install bitsandbytes
```

We define a  `flush(...)`  function to free all allocated memory to accurately measure the peak allocated GPU memory.

```python
import gc
import torch

def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
```

We can then load models in **8-bit quantization** by using a  `load_in_8bit=True`   flag, and measure the memory usage:

```python
del pipe
del model
flush()
model = AutoModelForCausalLM.from_pretrained("bigcode/octocoder", load_in_8bit=True, pad_token_id=0)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"
result = pipe(prompt, max_new_tokens=60)[0]["generated_text"][len(prompt):]
print(result)
```
Quantizing the model to 4-bit can be done with the same API as before - this time by passing `load_in_4bit=True` instead of `load_in_8bit=True`.


## Flash Attention

Self-attention layers in LLMs grow quadratically in compute and memory complexity with the number of input tokens. Flash Attention is an optimized algorithm that reduces memory costs while maintaining numerical accuracy.

**GPU Architecture Overview**

GPUs are composed of computational units (like floating-point units) and a memory hierarchy. Most modern GPUs include specialized low-precision matrix multiplication units (such as Nvidia's Tensor Cores for FP16/BF16 matrix multiplications).

The memory hierarchy in GPUs is divided into High Bandwidth Memory (HBM) and on-chip SRAM (also known as shared memory). For example, the A100 GPU features:

- **HBM**: 40-80GB with a bandwidth of 1.5-2.0TB/s.
- **SRAM**: Each of the 108 streaming multiprocessors shares 192KB, with a bandwidth of approximately 19TB/s.

GPU Memory Hierarchy image

### Flash Attention Idea 

**Flash Attention** is a technique widely used in mainstream large language models (LLMs). The main idea is to optimize the allocation of computations to fully utilize the GPU capabilities. Flash Attention is particularly beneficial for large language models that need to process long text input efficiently, e.g., for real-time applications where speed and responsiveness are crucial, such as chatbots and virtual assistants.

According to the Flash Attention paper, intermediate results of QKV (Query, Key, Value) computations are stored in SRAM instead of HBM. This reduces memory overhead to a linear level and achieves 2-4 times speedup by avoiding frequent read and write operations of intermediate results, thereby improving computational efficiency.

The Flash Attention v2 paper further reduces non-matrix multiplication operations and optimizes task distribution within the GPU to achieve significant speed improvements. 
- **Reduced Non-Matmul FLOPs**: By minimizing the floating-point operations that are not related to matrix multiplications.
- **Parallel Computation**: Tasks are distributed across different thread blocks for parallel computation, making full use of GPU resources.
- **Warp-Level Optimization**: Within a thread block, tasks are assigned to different warps to reduce shared memory access. 
  
These optimizations result in a 2-3 times speed improvement for Flash Attention v2. In their experiments, the metric used is TFLOPs/s, which stands for trillion floating-point operations per second. It's important to note that an increase in TFLOPs/s does not directly translate to a doubling of model throughput, i.e., tokens generated per second. Since Flash Attention optimizes the operations of self-attention (which are closely related to the input tokens), its effect is more noticeable with longer input sequences. When the input tokens are short, there is no significant speedup.  
 


  
**Implementation** 
Check the [official implementation](https://github.com/Dao-AILab/flash-attention) and [a simplified re-implementation](https://github.com/tspeterkim/flash-attention-minimal).

### Use of Flash attention

- Traditional inference  

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model = AutoModelForCausalLM.from_pretrained("bigcode/octocoder", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("bigcode/octocoder")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

system_prompt = """..."""  # Long system prompt
prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"

long_prompt = 10 * system_prompt + prompt

import time

start_time = time.time()
result = pipe(long_prompt, max_new_tokens=60)[0]["generated_text"][len(long_prompt):]
print(f"Generated in {time.time() - start_time} seconds.")
print(result)
```

- Enabling Flash Attention

```python
model.to_bettertransformer()

start_time = time.time()
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    result = pipe(long_prompt, max_new_tokens=60)[0]["generated_text"][len(long_prompt):]
print(f"Generated in {time.time() - start_time} seconds.")
print(result)
```


## Page Attention

Page Attention is a method designed to improve the management of memory, specifically the Key-Value (KV) cache, in large language models. This technique was inspired by virtual memory and paging in operating systems.

### The Challenge with KV Cache

-   **KV Cache**: In language models, the KV cache stores key and value pairs that the model uses to understand and generate text.
-   **Memory Consumption**: In models like LLaMA-13B, a single sequence's KV cache can take up to 1.7GB of memory. The size of this cache depends on the length of the input sequence, which can vary greatly and unpredictably.
-   **Inefficiency**: Managing this cache is challenging. Current systems often waste 60% to 80% of memory due to fragmentation (unused gaps in memory) and over-provisioning (reserving too much memory).

### How PagedAttention works

PagedAttention is an algorithm designed to manage the KV cache more efficiently. Here's how it works:

1.  **Inspiration from Paging**: PagedAttention takes inspiration from paging in operating systems. In computing, paging is a memory management scheme that eliminates the need for contiguous memory allocation by dividing memory into fixed-size blocks (pages).
    
2.  **Non-Contiguous Memory Storage**: Unlike traditional attention algorithms, PagedAttention stores the continuous key and value pairs in non-contiguous memory spaces. This means that instead of needing a big, uninterrupted chunk of memory, it can use smaller, scattered pieces.
    
3.  **Dividing the KV Cache**: The algorithm divides each sequence's KV cache into several blocks. Each block handles a fixed number of tokens (the smallest units of text, like words or characters).
    
4.  **Efficient Access**: During attention computation, PagedAttention can efficiently identify and access these blocks, improving memory utilization.
      
PagedAttention is integrated into [vLLM](https://github.com/vllm-project/vllm?tab=readme-ov-file), a library for serving large language models, making it easy to use.






## Improving Positional Embeddings of LLMs

To improve computational and memory efficiency in LLMs, we will explore improved Positional Embeddings.

### Understanding Positional Embeddings

Self-attention puts each token in relation to every other token. For instance, in the sequence “Hello”, “I”, “love”, “you”, each word token attends to all other word tokens with different probability masses, helping the model understand relationships between words. However, without positional embeddings, an LLM struggles to differentiate between different orders of words, such as "Hello I love you" and "You love I hello".

Positional embeddings encode the position of each token into a numerical representation, allowing the LLM to understand sentence order better. Types of Positional Embeddings: 

1. **Sinusoidal Positional Embeddings**: Introduced in the "Attention Is All You Need" paper, these embeddings are computed as sinusoidal functions of their positions.
2. **Learned Positional Embeddings**: Positional embeddings are learned during training rather than being fixed.

### Challenges with Absolute Positional Embeddings

Absolute positional embeddings encode a unique embedding for each position id (0, 1, ..., N), leading to poor performance on long text inputs and difficulty in extrapolating to input lengths longer than what the model was trained on.

### Relative Positional Embeddings

To address these issues, relative positional embeddings have become more popular, particularly:

1. **Rotary Position Embedding (RoPE)**
2. **ALiBi (Attention Linear Biases)**

#### Rotary Position Embedding (RoPE)

RoPE encodes positional information directly into query-key pairs by rotating each vector by an angle dependent on their position. This approach ensures that the probability score between query and key vectors depends on their relative distance rather than their absolute positions.

RoPE is used in models like:
- Falcon
- Llama
- PaLM

#### ALiBi (Attention Linear Biases)

ALiBi adds a negative integer scaled by a pre-defined value to each query-key entry right before the softmax computation, allowing the model to retain high performance on very long text input sequences.

ALiBi is used in models like:

- MPT
- BLOOM

### Advantages of Relative Positional Embeddings

1. **Extrapolation**: Both RoPE and ALiBi can extrapolate to input lengths not seen during training. ALiBi performs better out-of-the-box for extrapolated text inputs.
2. **Non-Learned Intuitions**: Both RoPE and ALiBi are based on intuitions that:
   - Positional cues should be given directly to the self-attention matrix.
   - The model should learn constant relative distance positional encodings.









## Key-Value Cache in Auto-Regressive Text Generation

Auto-regressive text generation with large language models (LLMs) involves iteratively providing an input sequence, sampling the next token, appending it to the sequence, and continuing until a stopping condition is met. This process can be optimized using key-value caches for efficiency.

### Auto-Regressive Text Generation

Auto-regressive generation works by iteratively generating tokens. The process involves:
1. Input a sequence.
2. Sample the next token using `torch.argmax`.
3. Append the sampled token to the input sequence.
4. Repeat until a stopping token is produced.

Example Code:

```python
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

for _ in range(5):
  next_logits = model(input_ids)["logits"][:, -1:]
  next_token_id = torch.argmax(next_logits, dim=-1)
  input_ids = torch.cat([input_ids, next_token_id], dim=-1)
  print("shape of input_ids", input_ids.shape)

generated_text = tokenizer.batch_decode(input_ids[:, -5:])
generated_text
```

### Using Key-Value Cache

LLMs are trained using the causal language modeling objective, which masks the upper triangle of the attention score matrix to ensure that tokens only attend to previous tokens. To reduce unnecessary computation, key-value vectors for all previous timesteps can be cached.  

In the following, we will tell the LLM to make use of the key-value cache by retrieving and forwarding it for each forward pass. In Transformers, we can retrieve the key-value cache by passing the  `use_cache`  flag to the  `forward`  call and can then pass it with the current token.

```python
past_key_values = None  # key-value cache
generated_tokens = []
next_token_id = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

for _ in range(5):
  next_logits, past_key_values = model(next_token_id, past_key_values=past_key_values, use_cache=True).to_tuple()
  next_logits = next_logits[:, -1:]
  next_token_id = torch.argmax(next_logits, dim=-1)

  print("shape of input_ids", next_token_id.shape)
  print("length of key-value cache", len(past_key_values[0][0]))  # [num_layers, 0 for k, 1 for v, batch_size, length, hidden_dim]
  generated_tokens.append(next_token_id.item())

generated_text = tokenizer.batch_decode(generated_tokens)
generated_text
```
As one can see, when using the key-value cache the text input tokens are  _not_  increased in length, but remain a single input vector. The length of the key-value cache on the other hand is increased by one at every decoding step. Making use of the key-value cache means that the $QK^T$ is essentially reduced to $q_c K^T$ with $q_c$ being the query projection of the currently passed input token which is always just a single vector.

Using a key-value (KV) cache in large language models (LLMs) can significantly enhance performance and efficiency, but it may not always be necessary or beneficial. Here are some objective guidelines on when to use a KV cache and when not to:

### When to Use KV Cache

1.  **Long Sequences**:
    
    -   **Efficiency**: For long input sequences or generating long outputs, using a KV cache can drastically reduce the computational load by avoiding repeated calculations for previously processed tokens.
    -   **Memory Management**: KV cache helps in managing memory efficiently by storing previously computed key-value pairs, preventing redundant computations.
2.  **Inference Speed**:
    
    -   **Real-Time Applications**: In scenarios where inference speed is critical, such as chatbots or interactive applications, the KV cache can help maintain low latency by speeding up token generation.
    -   **Batch Processing**: When processing large batches of inputs, using a KV cache can save significant time by reusing computations.
3.  **Resource Constraints**:
    
    -   **Limited Compute Resources**: If you are working with limited GPU/TPU resources, leveraging a KV cache can help make the most of the available hardware by reducing the amount of computation needed for each forward pass.

### When Not to Use KV Cache

1.  **Short Sequences**:
    
    -   **Overhead Management**: For very short sequences (e.g., those with fewer than 50 tokens), the overhead of managing the KV cache might outweigh the benefits, making it unnecessary.
    -   **Simple Tasks**: For simple tasks that do not require extensive computation, the KV cache might add complexity without significant gains in performance.
 
2.  **Model Training**:
    
    -   **Training Phase**: During the training phase of LLMs, the focus is on learning from the entire sequence. Using a KV cache is generally not applicable as each token's computation depends on all previous tokens in the training set.

3.  **Memory-Intensive Tasks**:
    
    -   **Memory Limits**: In scenarios where memory is extremely constrained and the added memory overhead of storing KV pairs could lead to out-of-memory errors, it might be better to avoid using a KV cache.
    -   **Memory Complexity**: For tasks where the memory complexity of the KV cache might introduce bottlenecks, careful consideration is needed to balance the trade-offs.




## Knowledge Distillation

Knowledge distillation is a technique where a smaller, simpler model (student) learns to replicate the behavior of a larger, more complex model (teacher). This process reduces the computational resources required for inference while maintaining similar performance levels.

### Process
1. **Train the Teacher Model**: Train a large, complex model on the target task.
2. **Generate Predictions**: Use the trained teacher model to generate predictions (soft labels) on the training data.
3. **Train the Student Model**: Train a smaller model using the soft labels from the teacher model, aiming to match the teacher's behavior.

### Advantages
- **Reduced Model Size**: The student model is smaller and less complex, making it more efficient.
- **Faster Inference**: The student model can process data faster, which is beneficial for real-time applications.
- **Resource Efficiency**: Lower computational and memory requirements.

### Usage in LLMs
Knowledge distillation is widely used in LLMs to create smaller, efficient models that retain most of the performance of their larger counterparts. This is particularly useful for deploying LLMs on edge devices or in environments with limited computational resources.

### Example
```python
import torch
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification

# Load pre-trained teacher and student models
teacher_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
student_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Example training loop for knowledge distillation
teacher_model.eval()
for batch in train_dataloader:
    inputs, labels = batch
    with torch.no_grad():
        teacher_outputs = teacher_model(**inputs)
    student_outputs = student_model(**inputs)
    
    # Distillation loss: compare student and teacher outputs
    distillation_loss = loss_fn(student_outputs.logits, teacher_outputs.logits)
    
    # Backpropagate and optimize the student model
    distillation_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Model Pruning

Model pruning involves removing redundant or less significant weights and neurons from the model to reduce its size and improve efficiency. There are two main types of pruning: unstructured and structured.

### Unstructured Pruning

-   **Removes Individual Weights**: Pruning individual weights that have minimal impact on model performance.
-   **Fine-Grained**: Provides fine control but can be harder to optimize for hardware acceleration.

### Structured Pruning

-   **Removes Entire Neurons or Filters**: Pruning whole neurons, channels, or filters in the network.
-   **Easier Hardware Optimization**: Structured pruning is more compatible with hardware optimization techniques and often leads to better performance gains.

### Advantages

-   **Reduced Model Size**: Significantly decreases the number of parameters, leading to smaller model sizes.
-   **Improved Inference Speed**: Less computation required, resulting in faster inference times.
-   **Maintained Performance**: Careful pruning can maintain or even improve model performance.

### Usage in LLMs

Pruning is used in LLMs to create leaner versions that are more efficient in terms of both memory and computation, without compromising much on accuracy. It is particularly useful in scenarios where computational resources are limited.

```python
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity

# Define a sample model
original_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Define pruning parameters
pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=0.5,
        begin_step=2000, end_step=10000
    )
}

# Apply pruning
pruned_model = sparsity.prune_low_magnitude(original_model, **pruning_params)

# Compile and train the pruned model
pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
pruned_model.fit(train_data, train_labels, epochs=10, callbacks=[sparsity.UpdatePruningStep()])
```

### References

-   [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
-   [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
-   [Flash Attention v2 Paper]()
-   [Dettmers et al.](https://arxiv.org/abs/2208.07339)
-   [GPTQ Paper](https://arxiv.org/abs/2210.17323)
-   [Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864)
-   [ALiBi (Attention Linear Biases)](https://arxiv.org/abs/2108.12409)


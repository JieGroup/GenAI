# Introduction

## Quick Survey
<iframe src="https://docs.google.com/forms/d/e/1FAIpQLScZsRUvrW6k9SWP2xKIFSNt3OYPgEAobyBnzboKxaIkRlKZng/viewform?embedded=true" width="640" height="880" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>

## About the Stat 8105 Course

Artificial Intelligence (AI) has evolved from a specialized field into a pivotal force shaping modern technology. Through our engagement with students across various disciplines, it is clear that many are intrigued by AI. They range from those engaged in PhD research to others seeking exceptional career opportunities or simply exploring their options. Despite their interest, a common challenge persists: a lack of holistic views about what to learn and how to effectively use AI tools. AI has experienced rapid advancements over the past two decades, with new methodologies and concepts emerging frequently. The terms "AI" and "Generative AI" themselves are broad, encompassing a wide range of disciplines including computer science, statistics, optimization, control, information theory, often leading to disparate interpretations and terminologies across different academic backgrounds. Fortunately, the main concepts and techniques of AI does not evolve as rapidly as it appears. Many thoughts are deeply rooted in established fields, such as how to organize data, set up the optimization problem, and interpretation of results. 

This Stat 8105 course is designed with the goal of lowing the barrier for new researchers and inspiring them to pursue state-of-the-art research in the relevant fields. We aim to achieve the goal through on-site implementations, explorations, discussions, and offline communications (as suggested in the syllabus). While many excellent open-source tutorials or courses exist, focusing primarily on practical implementation, our course will be research-oriented and encourage students to critically think the future directions they might take. 

**Acknowledgments**
This course was conceived and developed by Professor  [Jie Ding](https://jding.org/)  of the School of Statistics at the University of Minnesota. The [School of Statistics](https://cla.umn.edu/statistics) and the [Minnesota Supercomputing Institute (MSI)](https://msi.umn.edu/) are acknowledged for their administrative support and provision of computation and educational resources. Special thanks go to  [Xun Xian](https://jeremyxianx.github.io/), [Ganghua Wang](https://gwang.umn.edu/), Jin Du, An Luo, [Xinran Wang](https://wang8740.github.io/), Qi Le, Harsh Shah, [Jiawei Zhang](https://jiaweizhang.site/), [Enmao Diao](https://diaoenmao.com/), [Michael Coughlin](https://www.michaelwcoughlin.com/), and [Ali Anwar](https://chalianwar.github.io/), who provided significant contributions to the course design and development.


## Philosophy of Course Arrangement

Our discussions will often center around three fundamental elements:

- **Data**: The essential components organized and synthesized into actionable knowledge.
- **Learning**: The process through which we gain insights, predict outcomes, and develop generalizable knowledge from data.
- **AI**: The capability to structure data and learning processes autonomously.


|  | **Practice**                                                                                                                                                    | **Principles**                                                                                                                     |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| **Data**   | How to organize the available data? What kind of data processing is necessary?                                                                           | What piece of information is more relevant to the type of learning? How is the information effectively used?                        |
| **Learning**      | What are alternative algorithms and their practical considerations? Does the optimization convergence imply any generalizability at the population level? | What are alternative problem formulations? What is the rationale behind the learning problem?                                 |
| **AI**  | To what extent does the data + learning formulation exhibit autonomy? Can the developed ''AI'' survive in open-world environments where the underlying assumptions are not met? | Is this how human sets objectives when acting?                                                 |


Following the above framework, we refer to AI as the broad field concerned with creating systems that can perform tasks which would typically require human intelligence. Generative AI specifically refers to the focus on creating models that can generate new data samples by learning and approximating data distributions.


## How to Use This Course and Materials


### Prerequisite
This course is intended for research-oriented students. A basic understanding of statistics, machine learning, and programming experience in Python will be helpful.


### Expectation from Students
Students are expected to have:

1. Self-motivation to study independently and a significant amount of time dedicated to learning and homework assignments per week.

2. A willingness to engage in critical and creative thinking, to propose new research ideas, and to challenge existing practices.


### Outline of the Course

**Chapter 1. Kickoff and Quick Study of Deep Learning** (Sept 5, Sept 12)
Dive into the essentials of deep learning, exploring core model architectures such as CNNs, VAEs, and ResNets. Understand the computational underpinnings, including frameworks like Pytorch and Tensorflow, optimization methods like SGD and ADAM, and basic principles. We will set up the development environments, including setup of Python, Pytorch, IDE, Github, and optionally Minnesota Supercomputing Institute (MSI) and Huggingface. Finally, we will go through some (superficial) usage of pretrained large language models (LLMs), vision LLMs, text-to-image and text-to-video diffusion models.

**Chapter 2. Large Language Modeling** (Sept 12, 19)
We will dissect the model structures that underlie large language modeling, including various decoding algorithms and their inherent constraints, tokenizer training, positional encoding, attention mechanisms, and explore emergent linguistic abilities at scale such as zero-shot and few-shot learning capabilities.

**Chapter 3. Training LLMs from Scratch** (Sept 26)
We will cover the end-to-end process of building a transformer-based model from scratch. Specifically, we will go through computational tools such as Pytorch and cloud-computing, build model architectures, evaluate models, and examine how different hyperparameter choices could affect model performance.

**Chapter 4. Finetuning LLMs** (Oct 3)
This chapter introduces techniques for finetuning large language models to adapt them to specific tasks or user preferences. We will cover instruction finetuning, explore use cases such as LLM-based recommender systems and tabular data analysis, and examine the challenge of catastrophic forgetting during finetuning.


**Chapter 5. Human-Value Alignment** (Oct 10)
In this chapter, we will address the challenge of aligning LLM outputs with human values, such as making them more helpful, harmless, humorous, and personalized. We will discuss the technique of reinforcement learning from human feedback (RLHF) and explore emerging research directions focused on aligning AI systems with multi-dimensional value metrics.


**Chapter 6. Computation and Memory Problems in Foundation Models’ Training** (Oct 17)
We will target strategies for optimizing computation and memory usage in the training and finetuning of large foundational models. Topics include distributed training methodologies, memory-efficient techniques like ZeRO and parameter-efficient finetuning, alongside explorations of mixture-of-experts architectures.

**Chapter X. Project Preparation and Proposal Discussion** (Oct 17, Oct 24)

**Chapter 7. Diffusion Models** (Oct 31)
We will introduce diffusion models, a class of generative models that have shown remarkable proficiency in synthesizing high-quality data, especially text-to-image data. We will start with variation autoencoders and then discuss the architecture of many state-of-the-art diffusion models. We will explore how such models can enable the multi-scale representations crucial for generating photorealistic images from textual descriptions.

**Chapter 8. Agentic AI** (Nov 7)
We will explore the emerging field of agentic AI, focusing on autonomous agents capable of reasoning, planning, and acting in complex environments. This chapter will cover the mathematical foundations of agentic systems, including reinforcement learning, multi-agent systems, and planning algorithms. We will examine applications in autonomous systems, intelligent assistants, and scientific discovery, while addressing fundamental problems such as effectiveness and safety.

**Chapter 9. Retrieval Augmented Generation** (Nov 14)
We will examine how Retrieval-Augmented Generation (RAG) can improve the performance of LLMs in knowledge-intensive tasks, examining its computational scalability and safety.

**Chapter 10. Efficient Deployment Strategies for Foundation Models** (Nov 21)
We will introduce standard techniques to reduce sizes of standard deep model and transformer models, such as compiling, quantization, pruning, knowledge distillation, their applications to model deployment, and the statistical rationales.

**Chapter 11. Ethics and Safety in Generative AI** (Nov 21)
As we navigate the complexities and capabilities of AI technologies, ethical considerations will form an integral part of our curriculum. We will introduce quantitative methods and metrics to assess the safety of generative models from two perspectives. One is from an ethics perspective, including fairness and toxicity. We will also delve into content moderation techniques grounded in statistical detection and watermarking techniques. The other is from a machine learning security perspective. We will study several angles that practical AI systems must counteract, including adversarial examples, privacy, data poisoning backdoor attacks, membership inference attacks, model-stealing attacks, and their statistical foundations. The lecture will also discuss how these security issues are arising in large generative models.

**Chapter Y. Final Project Presentation and Discussion** (Dec 5)
In summary, this course is not an introduction of generative AI tools. It is a research-oriented course designed for PhD students to quickly learn the key toolsets and principles that drive the field. By offering hands-on experiences and highlighting theoretical bases, we aim to equip students to apply their knowledge across scientific domains and foster innovative AI-for-science initiatives.


### How to Gain from the Course

This website contains the main lecture notes created by GenAI course team. Each chapter will consist of background of the problem, formulations, principles, and implementation strategies. In line with the motivation and philosophy of the course design, the course will be a highly interweaved mixture of practical coding, study of basic principles, and open problems raised to stimulate further thinking. Specifically, we will have a significant portion of the course material dedicated to code implementation so you can immediately apply the learned techniques to various domain problems. 

**Software Environment**
Python is chosen for its versatility and simplicity. It is widely used in the AI community due to its extensive libraries and community support. Detailed instructions on setting up this environment will be covered at the end of this introduction., Necessary dependencies will be provided to ensure you can seamlessly transfer code into a Jupyter notebook within each chapter.  

**Hardware Environment** 
Thanks to the generous support from Minnesota Supercomputing Institute (MSI), all enrolled students will receive an educational account providing access to a GPU cluster equipped with a Slurm fair queuing system. This setup allows you to utilize necessary computational resources to practice the techniques learned and to conduct research projects related to the course.

### (One-time) Quick Setup of Python Environment

This guide provides step-by-step instructions on how to create a Python virtual environment named `GenAI` using Python 3.8, and then how to install packages using a `requirements.txt` file.

#### Approach One: Using Python's Built-in `virtualenv` Tool 

This approach has a faster setup and takes less disk space compared with Approach Two, ideal for those who need a lightweight setup.

**Step 1: Install Python**: Ensure that Python 3.8 or above is installed on your system. You can download it from [python.org](https://www.python.org/downloads/release/python-380/). Alternatively, use command line as follows. On Windows, you can search for Command Prompt in the Start menu. On macOS or Linux, open the Terminal application.

- On Windows: first install Chocolatey (a package manager for Windows) and then can install Python using
     ```bash
	choco install python --version=3.8
     ```
- On macOS or Linux: first install Homebrew (a popular package manager) and then install Python using
   ```bash
	brew install python@3.8
   ```

**Step 2: Create a Virtual Environment**: Navigate to your project directory, and create a virtual environment named `GenAI` using Python 3.8.

   - On Windows:
     ```bash
     cd path/to/your-project
     py -3.8 -m venv GenAI
     ```

   - On macOS or Linux:
     ```bash
     cd path/to/your-project
     python3.8 -m venv GenAI
     ```

**Step 3: Activate the Virtual Environment**

- On Windows:
  ```bash
  .\GenAI\Scripts activate
  ```

- On macOS or Linux:
  ```bash
  source GenAI/bin/activate
  ```

Once activated, your terminal prompt should change to indicate that you are now working within the `GenAI` virtual environment. You have now successfully created a Python environment named `GenAI`, which is isolated from other Python environments on your system. To deactivate the environment and return to your global Python environment, run:
```bash
deactivate
```

**Step 4: Install Packages**: Install one or more packages using
   ```bash
   pip install package_name1 package_name2 ...
   ```
If you have a `requirements.txt` file that lists all the packages you want to install along with their versions, you can batch-wise install them by running
   ```bash
   pip install -r requirements.txt
   ```

#### Approach Two: Using Conda

This approach uses Anaconda or Miniconda for a more integrated environment management. Conda commands are identical across operating systems.

**Step 1: Install Anaconda or Miniconda**: For a full-featured environment with numerous pre-installed packages, [download Anaconda](https://www.anaconda.com/products/distribution). For a more minimal installation, [download Miniconda](https://docs.conda.io/en/latest/miniconda.html).
   
**Step 2: Create a New Conda Environment**
   ```bash
	conda create --name GenAI python=3.8
   ```
   
**Step 3: Activate or Deactivate the Conda Environment**
   ```bash
	conda activate GenAI
	conda deactivate
   ```

**Step 4: Install Packages**: Install one or more packages using
   ```bash
	pip install package_name1 package_name2 ...
   ```
For our course, you will likely need the following packages:
   ```bash
	pip install accelerate datasets diffusers gym huggingface-hub ipykernel matplotlib numpy opencv-python pandas transformers 
   ```
If you have a `requirements.txt` file that lists all the packages you want to install along with their versions, you can batch-wise install them by running
   ```bash
   pip install -r requirements.txt
   ```

For Jupyter Notebook users, to ensure the newly created Conda environment named `GenAI` shows up in Jupyter Notebook, you need to install the `ipykernel` package in that environment and then register it with Jupyter. 
```bash
pip install notebook ipykernel
python -m ipykernel install --user --name GenAI --display-name "Python 8 (GenAI)"
jupyter notebook
```
For MSI ondemand cloud users, remember to add the following lines as attached when launching a jupyter in ondemand, so you access the right Python path once activating `GenAI`.
```bash
module load conda
module load cuda
```

### Additional Resources

Here are free resources to help you develop skills for conducting advanced R&D. While Python and PyTorch are required, familiarity with other technologies listed will greatly enhance your learning experience and capability to address practical challenges.

#### Learn Python

Python is a fundamental skill for anyone working in data science and AI. Here are some great resources to get started or to sharpen your Python skills:

-   [Python Official Tutorial](https://docs.python.org/3/tutorial/index.html) - Ideal for beginners.
-   [Real Python](https://realpython.com) - In-depth tutorials and explanations on various Python topics.
-   [Automate the Boring Stuff with Python](https://automatetheboringstuff.com) - Fun approach to learning Python.

#### Learn Pytorch

PyTorch is one of the leading libraries for deep learning research. Here's where you can learn more about it:

Install it immediately after setting up the Python environment!
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # customized based on https://pytorch.org/get-started/locally/
```
-   [PyTorch Official Tutorials](https://pytorch.org/tutorials/) - Comprehensive tutorials from the official PyTorch website.
-   [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) - A quick introduction to the core concepts of PyTorch.

#### Learn CUDA

CUDA is a parallel computing platform and application programming interface model created by Nvidia. It allows software developers to use a CUDA-enabled graphics processing unit (GPU) for general-purpose processing. Here are some tutorials to get started:

-   [CUDA Tutorials on PyTorch](https://pytorch.org/tutorials/advanced/cpp_extension.html) - Learn how to extend PyTorch with custom C++ and CUDA extensions.
-   [GPU Puzzles](https://github.com/srush/GPU-Puzzles) - Practice your GPU programming skills with these puzzles.

#### Learn Slurm

Slurm is a highly configurable open-source workload manager. Here are some resources to learn more about managing computing workloads:

-   [Job submission and scheduling with Slurm](https://msi.umn.edu/our-resources/knowledge-base/slurm-job-submission-and-scheduling) - MSI-provided guideline of using Slurm.

#### Learn Kubernetes

Kubernetes is an open-source system for automating deployment, scaling, and management of containerized applications. 
-   [Kubernetes documentations](https://kubernetes.io/docs/tutorials/) - Kubernetes basics.
-   [Kubernetes Course - Full Beginners Tutorial](https://www.youtube.com/watch?v=X48VuDVv0do) - A video tutorial for beginners.


## Course Survey
<iframe src="https://docs.google.com/forms/d/e/1FAIpQLScYzVRFj4sMBZCK6qr_4-K0f_TYLDt4cY2NczyzSRYO7W5sbA/viewform?embedded=true" width="640" height="1408" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>
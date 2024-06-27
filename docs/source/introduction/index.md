# Introduction

## The Surge of AI: The Motivation for This Course

Artificial Intelligence (AI) has evolved from a specialized field into a pivotal force shaping modern technology. Through our engagement with students across various disciplines, it's clear that many are intrigued by AI. They range from those engaged in PhD research to others seeking exceptional career opportunities or simply exploring their options. Despite their interest, a common challenge persists: a lack of confidence in their AI expertise, leading to uncertainty about what to learn and how to effectively use AI tools. This confusion isn't unwarranted; AI has experienced rapid advancements over the past two decades, with new methodologies and concepts emerging almost daily. The terms "AI" and "Generative AI" themselves are broad, encompassing a wide range of disciplines including statistics, optimization, control, information theory, and even hardware and systems design, often leading to disparate interpretations and terminologies across different academic backgrounds.

Fortunately, the core concepts and techniques of AI does not evolve as rapidly as it appears. Many thoughts are deeply rooted in established fields, such as how to organize data, set up the optimization problem, and interpretation of results. Just like a [make a story here simply] building a car, we are changing the engine, appearing, and material, but the design principle is often the same.  As such, our course  is designed for both new and experienced researchers. 

Despite the apparent rapid evolution, the foundational concepts and techniques of AI remain relatively stable, much like the underlying principles of car manufacturing. While the components like engines and materials may change, the fundamental design principles persist. Our course caters to both newcomers and seasoned researchers, aiming to demystify AI through both hands-on coding experiences and robust understanding of its core principles.

**Purpose and Audience**: 
- **Purpose**: To establish a solid foundation in Generative AI, promoting both critical and creative thinking.
- **Audience**: This course welcomes anyone interested in AI, ranging from novices to experienced researchers.

While several open-source AI courses exist, focusing primarily on practical implementation, our course emphasizes the rationale behind AI techniques and the future directions they might take. This approach aims to empower students with not only the skills to apply AI concepts but also the ability to innovate and advance their research.

### Philosophy of Course Arrangement: A Triple-Dimension Framework


To clarify the sometimes ambiguous terms "AI" and "Generative AI," our course introduces a structured framework comprising three fundamental elements:

- **Information**: The essential components organized and synthesized into actionable knowledge.
- **Learning**: The process through which we gain insights, predict outcomes, and develop generalizable knowledge from data.
- **Intelligence**: The capability to structure data and learning processes autonomously.

| **Triple-Dimension** | **Practice**                                                                                                                                                    | **Principles**                                                                                                                     |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| **Information**   | How to organize the available data? What kind of feature engineering is necessary?                                                                           | What piece of information is more relevant to the type of learning? How is the information effectively used?                        |
| **Learning**      | What are alternative optimization algorithms and their practical considerations? Does the optimization convergence imply any generalizability at the population level? | What are alternative problem formulations? What is the rationale behind the learning problem?                                 |
| **Intelligence**  | To what extent does a data+learning formulation exhibit autonomy? Can the developed ''AI'' survive in open-world environments where the underlying assumptions are not met? | Is this how human sets objectives when acting? Does it reflect any level of ''true intelligence''?                                                 |


Following this Triple-Dimension Framework, we refer to AI as the broad field concerned with creating systems that can perform tasks which would typically require human intelligence. Generative AI specifically refers to the focus on generative modeling, which involves creating models that can generate new data samples by learning and approximating data distributions.


## How to Use This Course and Materials


### Prerequisite
This course does not have strict graduate-level prerequisites. However, a basic understanding of statistics or machine learning, along with some programming experience (not necessarily in Python), will be advantageous. This background will allow you to focus more on critical thinking and research development, optimizing the time spent on each chapter.

### Expectation from Students
Students are expected to demonstrate:

1. A keen interest in AI and self-motivation to study independently. While each chapter is designed to be nearly independent, a full grasp of the material will likely require 10 hours of study per week.

2. A willingness to engage in critical and creative thinking, to propose new research ideas, and to challenge existing practices.


### Outline of the Course


**Chapter 1. Quick Study of Deep Learning**
Dive into the essentials of deep learning, exploring core model architectures such as CNNs, VAEs, and ResNets. Understand the computational underpinnings, including frameworks like Pytorch and Tensorflow, optimization methods like SGD and ADAM, and foundational principles. We'll also showcase diverse applications, spanning natural language processing to computer vision.  

**Chapter 2. Large Language Modeling**
Language modeling's trajectory from N-gram models through Word2vec to contemporary pre-trained models will be charted. We will dissect the probabilistic methodologies that underlie language modeling, including various decoding algorithms and their inherent constraints. We will also dissect the transformer architecture to understand its core components, including attention mechanisms, positional encoding, and the significance of self-attention over recurrent or convolutional layers. We will then examine the myriad of transformer variants that have emerged and evaluate the empirical observations that have been made about transformer models, such as their surprising ability to perform zero-shot learning, their emergent linguistic abilities at scale, and their application in few-shot learning scenarios. The lecture will also touch upon the interpretability of these models, discussing how attention maps can offer insights into model decisions and the representation of language concepts.

**Chapter 3. Training LLMs from Scratch**
This lecture will introduce standard steps and techniques used in training LLMs from scratch, addressing tokenizer training, self-supervised pre-training, and instruction finetuning. The lecture will also connect the algorithms and fundamental principles in optimization and statistical learning.

**Chapter 4. Human-Value Alignment**
Explore the critical aspect of human-value alignment in AI development, discussing the ethical and practical implications of AI decisions and actions.

**Chapter 5. Diffusion Models**
This lecture will introduce diffusion models, a class of generative models that have shown remarkable proficiency in synthesizing high-quality data, especially text-to-image data. We begin by laying the theoretical groundwork, explaining the stochastic diffusion process that gradually adds noise to data and the reverse process that generates new data from noise. We will then discuss the architecture of UNet models, the backbone of many diffusion processes, detailing how their design enables the capture of multi-scale representations crucial for generating photorealistic images from textual descriptions. The lecture will also cover the latest innovations aimed at improving efficiency, such as the integration of autoencoder techniques for more resource-efficient training and generation.

**Chapter 6. Computation and Memory Problems in Foundation Models’ Training**
This lecture targets strategies for optimizing computation and memory usage in the training and finetuning of large foundational models. Topics include distributed training methodologies, memory-efficient techniques like ZeRO, Flash-Attention, and parameter- efficient finetuning, alongside explorations of mixture-of-experts architectures.

**Chapter 7. Efficient Deployment Strategies for Foundation Models**
This lecture will introduce standard techniques to reduce sizes of standard deep model and transformer models, such as quantization, pruning, knowledge distillation, their applications to model deployment, and the statistical rationales.

**Chapter 8. Retrieval Augmented Generation**
Examine how Retrieval-Augmented Generation (RAG) can improve the performance of LLMs in knowledge-intensive tasks, examining its computational scalability and safety.

**Chapter 9. Safety in Generative AI**
This lecture will introduce quantitively methods and metrics to assess the safety of generative models from two perspectives. One is from ethics perspective, including fairness and toxicity. We will also delve into content moderation techniques grounded in statistical detection and state-of-the-art watermarking techniques.
The other is from machine learning security perspective. We will study several angles that practical AI systems must counteract, including adversarial examples, privacy, data poisoning backdoor attacks, membership inference attacks, model-stealing attacks, and their statistical foundations. The lecture will also discuss how these security issues are arising in large generative models.

**Chapter 10. Research Projects: Application Cases or New Methods**
We will study student-identified research problems, such as application cases or new methodological development. 

### How to Benefit Most from This course

This website contains the main lecture notes created by GenAI course team. Each chapter will consist of background of the problem, formulations, principles, and implementation strategies. In line with the motivation and philosophy of the course design, the course will be a highly interweaved mixture of practical coding, study of basic principles, and open problems raised to stimulate further thinking. Specifically, we will have a significant portion of the course material dedicated to code implementation so you can immediately apply the learned techniques to various domain problems. 

This website is home to the comprehensive lecture notes crafted by Prof. Jie Ding for the GenAI course. Each chapter is structured to cover the background of the topic, problem formulations, underlying principles, and practical implementation strategies. Reflecting the course's philosophy, the content is an integrated blend of hands-on coding exercises, theoretical exploration, and discussions on open problems designed to encourage further inquiry and innovation. 

**Software Environment**
A substantial portion of the course is devoted to coding implementations, allowing you to apply the techniques you learn directly to problems across various domains. Python is chosen for its versatility and simplicity, making it ideal for both beginners and experienced programmers. It is widely used in the AI community due to its extensive libraries and community support.

All course materials have been developed for the Python 3.8 environment. Detailed instructions on setting up this environment  will be covered at the end of this introduction., Necessary dependencies will be provided to ensure you can seamlessly transfer code into a Jupyter notebook within each chapter.  

**Hardware Environment**
Thanks to the generous support from MSI at UMN, we will have educational account for all participants of the class. Students can access GPU cluster and slurm fair queueing system to access computation resources needed to practice the learned stuff and accompllish course-related research projects. 
 
Thanks to the generous support from Minnesota Supercomputing Institute (MSI), all UMN course participants will receive an educational account providing access to a GPU cluster equipped with a Slurm fair queuing system. This setup allows you to utilize necessary computational resources to practice the techniques learned and to conduct research projects related to the course.
 
### (One-time) Quick Setup of Python Environment

This guide provides step-by-step instructions on how to create a Python virtual environment named `GenAI` using Python 3.8, and then how to install packages using a `requirements.txt` file.

#### Approach One: Using Python's Built-in `virtualenv` Tool 

This approach has a faster setup and takes less disk space compared with Approach Two, ideal for those who need a lightweight setup.

**Step 1: Install Python**: Ensure that Python 3.8 is installed on your system. You can download it from [python.org](https://www.python.org/downloads/release/python-380/). Alternatively, use command line as follows. On Windows, you can search for Command Prompt in the Start menu. On macOS or Linux, open the Terminal application.

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
  .\GenAI\Scriptsctivate
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
   conda install package_name1 package_name2 ...
   ```
If you have a `requirements.txt` file that lists all the packages you want to install along with their versions, you can batch-wise install them by running
   ```bash
   pip install -r requirements.txt
   ```
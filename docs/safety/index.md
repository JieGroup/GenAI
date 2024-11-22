# 10. AI Safty
 

## Poisoning Backdoor Attacks
    

As machine learning models become integral to various applications, their security emerges as a critical concern. In real-world scenarios, these models are often constructed from datasets whose intricacies may be obscured from users. This lack of transparency poses a risk for exploitation through **backdoor attacks**, a growing concern in AI security. These attacks are designed to make a model operate normally until it encounters specific, altered inputs that activate the backdoor, causing the model to behave unpredictably, as demonstrated in the following figure.

  
**Figure: An example of a backdoor attack that compromises the traffic sign classifier for autonomous driving.**
<div style="text-align:center; margin-bottom: 50px;">
    <img src="../_static/img/fig_stopsign.png" alt="Sample Aug" width="300" style="display:block; margin:auto;">
</div>
  

A powerful backdoor attack has a **dual-goal**: being stealthy and useful, meaning that it 
- prompt the compromised model to exhibit manipulated behavior when a specific attacker-defined trigger is present, and 
- maintain normal functionality in the absence of the trigger, rendering the attack difficult to detect.

  
### Demystifying Poisoning Backdoor Attacks from a Statistical Perspective

Recent research aims to tackle the following crucial yet previously underexplored questions:

1. What are the key factors determining backdoor attack’s success?

2. What shape or direction of a trigger signal constitutes the most potent backdoor while maintaining the same level of utility distortion?

3. When will a human-imperceptible trigger succeed?

  

To address these questions, recent research has quantitatively revealed three key factors that jointly determine the performance of any backdoor attack: 
- **the ratio of poisoned data $\rho$**, 
- **the direction and magnitude of the trigger $\eta$**, and 
- **the clean data distribution $\mu$**, as shown below.
  
  
**Figure: Illustration of three factors jointly determining the effectiveness of a backdoor attack: poisoning ratio, backdoor trigger, and clean data distribution.**
<div style="text-align:center; margin-bottom: 50px;">
    <img src="../_static/img/fig_backdoor_illustrate.png" alt="Sample Aug" width="650" style="display:block; margin:auto;">
</div>

   
  

Existing research also quantified the prediction performance, denoted by $r_n$, of a backdoored model on both clean or backdoored data through a finite-sample analysis. Briefly speaking, the team has shown

$$
r_n \sim g(\rho,\eta,\mu)
$$

where $g(\cdot)$ is an explicit function delineating the prediction performance's dependence on three principal factors. This analytical framework is **applicable to both discriminative and generative models**. More technical details can be found in [this paper](https://openreview.net/pdf?id=BPHcEpGvF8).


The above result then implies answers to the last two questions:

- The **optimal trigger** direction is where the clean data distribution decays the most.

- Constructing a **human-imperceptible backdoor attack could be more feasible** when the clean data distribution degenerates more.


The above fundamental understanding also serves as a basis for developing improved defense mechanisms against backdoor attacks.
  
 
  

### Mitigating Backdoor Attack
    
  

The hidden nature of backdoor attacks underscores the urgency of developing robust defenses. Current strategies fall into two categories: inference-stage defenses for detecting backdoored data at the point of use, and training-stage defenses to prevent neural networks from learning from backdoored training data. As Machine Learning as a Service (MLaaS) becomes increasingly common, there is a growing need for real-time, on-the-fly defense mechanisms against backdoor attacks. In such a context, inference-stage defenses are critical because they offer the last line of defense, operating at the point where the model is actually used to make predictions on new data. However, they often lack theoretical foundation and are typically limited to vision tasks, leaving a gap in natural language processing applications.


**Figure: Illustration of inference-stage backdoor defenses. The backdoor triggers in text queries are indicated in red, while for the image queries, the backdoor triggers consist of a square patch located in the lower-right corner for the traffic sign and a hello kitty embedding added to an image of a dog.**
<div style="text-align:center; margin-bottom: 50px;">
    <img src="../_static/img/fig_backdoor_defenses.png" alt="Sample Aug" width="650" style="display:block; margin:auto;">
</div>

 

A recent framework is known as Conformal Backdoor Detection (CBD). This framework is tailored to combat backdoor attacks by adeptly pinpointing query inputs that have been tampered with by adversaries. CBD establishes a new benchmark that achieves the state-of-the-art backdoor detection accuracy in the field. What distinguishes CBD further is its ability to provide empirical guarantees concerning the False Positive Rate (FPR). This capability ensures a quantifiable level of reliability in distinguishing genuine samples from those compromised by backdoor manipulations.

  

Formally, given a backdoored model and a clean validation dataset, upon receiving a test query $X_{\text{test}} \in \mathbb{R}^d$, the defender sets a hypothesis testing problem:

$$
\texttt{H}_0: T(X_{\text{test}}) \sim T(S); \quad S \sim \mathbb{P}_{\textrm{CLEAN}}, \quad
\texttt{H}_1: T(X_{\text{test}}) \not\sim T(S); \quad S \sim \mathbb{P}_{\textrm{CLEAN}},
$$

where $T(\cdot)$ is a latent representation of query input $X_{\text{test}}$ under the backdoored model. The use of $T(\cdot)$ is to reduce the dimensionality of data $X_{\text{test}}$, e.g., images and texts.

Since the **backdoor data distribution, $\mathbb{P}_{\text{BD}}$, is unknown to the defender** in practice, defenders will construct a detector specified by

$$
g(X ; s, \tau)=\left\{\begin{array}{ll}
1 \quad (\text{Backdoor-triggered Sample}), & \textrm{ if } s(T(X)) \geq \tau \\
0 \quad (\text{Clean Sample}), & \textrm{ if } s(T(X))<\tau
\end{array}\right.
$$


where $\tau \in \mathbb{R}$ is a threshold value and $s(\cdot)$ is a scoring function indicating the chance of $X_{\text{test}}$ being a clean input.

The defender aims to devise a detector \(g\) to

$$
\text{maximize } \quad \mathbb{P}\bigl\{ g(X ; \tau) = 1 \mid X \text{ is backdoor} \bigr\}, \\ 
\text{while controlling the false positive rate (FPR):} \quad 
\mathbb{P}\bigl\{ g(X ; \tau) = 1 \mid X \text{ is clean} \bigr\}.
$$

To effectively mitigate backdoor attacks on Deep Neural Networks, the newly proposed methodology employs a conformal prediction framework to precisely control the False Positive Rate (FPR). By leveraging a decision threshold based on empirical data distribution, this approach remains statistically rigorous without depending on explicit distribution assumptions.
  

### Jailbreak Attacks

Jailbreak attacks represent a form of inference-stage vulnerability in AI models, analogous to backdoor attacks at the training stage. These attacks manipulate an AI system to bypass its safety constraints, generating unintended or harmful outputs when given crafted prompts.


**Figure: While backdoor attacks require tampered training data, jailbreak attacks operate during inference and target vulnerabilities in the model's prompt handling. [image source](https://arxiv.org/pdf/2401.09002v2)**
<div style="text-align:center; margin-bottom: 50px;">
    <img src="../_static/img/safety_jailbreak.png" alt="Sample Aug" width="300" style="display:block; margin:auto;">
</div>






## Model Stealing Attacks and Defenses

AI applications often involve interactions between service providers and users. For instance, a provider (Alice) might deploy a proprietary model through an API that takes a user's query (Bob's input) and returns a response. This model could represent anything from a finely tuned deep learning system to a physical simulation engine. While such interactions are central to Machine-Learning-as-a-Service (MLaaS) systems, they also create vulnerabilities.

If Bob collects enough input-output pairs by querying Alice's model, he could reconstruct the model, compromising its intellectual property. This risk gives rise to **model stealing attacks**, where adversaries aim to infer or replicate Alice's model through repeated queries. Addressing these risks requires balancing the utility of the model for legitimate users with enhanced privacy safeguards against adversaries.

 
### **Mechanisms of Model Stealing**

- **Supervised Reconstruction**: Adversaries use input-output pairs as supervised training data to replicate the model.
- **Query Optimization**: Efficient query strategies maximize information gain from the model while minimizing the number of queries required.
- **Gradient Exploitation**: In cases where gradient information is accessible, adversaries can leverage it to accelerate model reconstruction.

**Toy Case**: Consider a linear model that returns $x^T \beta$ for each query $x $, where $\beta \in \mathbb{R}^d$. How many queries are needed to reconstruct $\beta$?


### **Defenses Against Model Stealing**

Key questions arise in this context:
- **How can model privacy be enhanced for already-learned models?**
- **What are the tradeoffs between model privacy and utility?**

Defensive strategies aim to deter adversaries while preserving the model's utility for legitimate users. These strategies fall into the following categories:


**1. Query Monitoring**

- **Information Gain Analysis**: Monitor the information gain of queried inputs and flag suspicious patterns that deviate from typical user query.

- **Statistical Anomaly Detection**: Use distributional tests to identify adversarial queries based on their input-output characteristics.



**2. Jointly Perturb Input and Output**:

Introduce controlled noise to user queries, ensuring the model receives modified inputs that obscure its true functionality.

For example, the **information laundering** framework provides a mechanism to co-design input and output transformations that balance privacy and utility from an information-theoretic perspective.



**Information Laundering Framework**

**Definition**:  A learned model is a kernel \( p: \mathcal{X} \times \mathcal{Y} \rightarrow [0,1] \), which induces a class of conditional distributions \( \{p(\cdot \mid x): x \in \mathcal{X}\} \).

**Definition**:  An information-laundered model with respect to a given model \( \mathcal{C} \) is a model \( \mathcal{C}' \) that consists of three internal kernels:

$$
\mathcal{C}' = \mathcal{A} \circ \mathcal{C} \circ \mathcal{B},
$$

where $\mathcal{A}$ and $\mathcal{B}$ are input and output perturbation kernels, respectively. This is illustrated in the figure below.



**Figure: Illustration of Information Laundering (a) Alice's effective system for public use, and (b) Alice's idealistic system not for public use.** [source image](https://openreview.net/pdf?id=dyaIRud1zXg)
<div style="text-align:center; margin-bottom: 50px;">
    <img src="../_static/img/safety_laundering.png" alt="Sample Aug" width="600" style="display:block; margin:auto;">
</div>

We denote the kernels representing the authentic model, input kernel, output kernel, and the information-laundered model as $p(\cdot \mid \cdot), p_{\mathcal{A}}(\cdot \mid \cdot), p_{\mathcal{B}}(\cdot \mid \cdot), p_{\mathcal{C}}(\cdot \mid \cdot)$, respectively.

The information laundering objective minimizes the following function:

$$
L(p_{\mathcal{A}}, p_{\mathcal{B}}) =
\mathbb{E}_{X \sim p_X} \mathcal{D}(p(\cdot \mid X), p_{\mathcal{C}}(\cdot \mid X)) + \beta_1 \mathcal{I}(X; \mathcal{X}) + \beta_2 \mathcal{I}(Y; \mathcal{Y}),
$$

where:
- $\mathcal{D}$ represents the divergence between the true model and the laundered model.
- $\mathcal{I}$ represents mutual information to control privacy leakage between original and perturbed variables.

Using the **calculus of variations**, the framework derives closed-form optimal perturbation kernels for both inputs and outputs, enabling practical implementations of privacy-utility tradeoffs.






## **References**

- Alabbadi, M. M. "Mobile cloud computing: A revolution in computing and communication." *Proceedings of the International Conference on Mobile Cloud Computing, Services, and Engineering*. 2011.
- Ribeiro, M. T., Singh, S., & Guestrin, C. "Why should I trust you? Explaining the predictions of any classifier." *Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. 2016.
- Kesarwani, M., et al. "Model extraction warning in MLaaS platforms." *Proceedings of the ACM SIGKDD International Conference*. 2018.
- Juuti, M., et al. "PRADA: Protecting against DNN model stealing attacks." *Proceedings of the IEEE Symposium on Security and Privacy*. 2019.




## Watermarking 

The rapid advancements in Generative AI, particularly in diffusion models (DM) like **Stable Diffusion** and **DALLE-2**, have significantly improved image generation quality. However, these innovations raise concerns, including **DeepFake** misuse and **copyright infringement**. To mitigate misuse, **watermarking** techniques offer a way to identify machine-generated content by embedding distinct signals.

### Watermarking Techniques

Watermarking for generative models can be categorized into two groups:

1. **Model-Specific Methods**: Tailored for specific generative models, offering improved trade-offs between watermark quality and detection performance. For example, [Tree-Ring](https://proceedings.neurips.cc/paper_files/paper/2023/file/b54d1757c190ba20dbc4f9e4a2f54149-Paper-Conference.pdf) works with DDIM samplers.

2. **Model-Agnostic Methods**: Apply watermarks directly to generated content without modifying the models. These include:
   - **Traditional Techniques**: Embed signals in the image frequency domain, e.g., DwTDcT, but are vulnerable to strong manipulations.
   - **Deep Learning Techniques**: Use encoder-decoder architectures to embed robust watermarks, e.g., [RivaGan](https://arxiv.org/pdf/1909.01285). These methods are computationally intensive, making real-time deployment challenging.


Watermarking strategies are evaluated based on false-positive rates (FPRs) and the Area Under the Receiver Operating Characteristic curve (AUROC). However, they are empirically found to be not robust against adversarial manipulations or unforeseen distribution shifts in real-world scenarios. 


### **R**obust, **A**gile, and plug-and-play **W**atermarking (**RAW**)

Recent work RAW introduces a real-time, model-agnostic framework for watermarking generative content, with the following design elements:

- **Dual-Domain Embedding**: Watermarks are embedded in both frequency and spatial domains for enhanced robustness:
  $$
  \mathcal{E}_{\boldsymbol{w}}(X) = \mathcal{F}^{-1}(\mathcal{F}(X) + c_1 \cdot v) + c_2 \cdot u,
  $$
  where $v, u$ are watermarks in frequency and spatial domains, and $c_1, c_2$ control visibility.
- **Efficiency**: Batch processing makes watermark injection up to $30 \times$ faster than frequency-based methods and $200 \times$ faster than encoder-decoder methods.
- **Robustness**: Adapts adversarial training and contrastive learning for improved resilience.

**Figure: While backdoor attacks require tampered training data, jailbreak attacks operate during inference and target vulnerabilities in the model's prompt handling.** [source image](https://openreview.net/pdf?id=ogaeChzbKu)
<div style="text-align:center; margin-bottom: 50px;">
    <img src="../_static/img/safety_watermark.png" alt="Sample Aug" width="600" style="display:block; margin:auto;">
</div>


**Problem Formulation**

Watermarking is formalized as a binary classification problem:
$$
\texttt{H}_0: \widetilde{X}^{\prime} \text{ is watermarked}; \quad \texttt{H}_1: \widetilde{X}^{\prime} \text{ is not watermarked.}
$$

A detector is defined as:
$$
g(\mathcal{V}_\theta(X)) =
\begin{cases}
1 \quad (\text{Watermarked}), & \text{if } \mathcal{V}_\theta(X) \geq \tau, \\
0 \quad (\text{Unwatermarked}), & \text{if } \mathcal{V}_\theta(X) < \tau,
\end{cases}
$$
where $\mathcal{V}_\theta(X)$ scores the likelihood of watermark presence, and $\tau$ is a threshold.



**Training Stage**

   - **Watermarking Module**: Embeds watermarks into images using spatial and frequency transformations.
   - **Verification Module**: Binary classifier scoring the likelihood of watermarked images.

   The combined training loss incorporates original and augmented datasets:
   $$
   \mathcal{L}_{\text{raw}} = \operatorname{BCE}(\mathcal{D}) + \sum_{k=1}^m \operatorname{BCE}(\mathcal{D}^k),
   $$
   where $\operatorname{BCE}$ is the binary cross-entropy loss and $\mathcal{D}^k$ are augmented datasets.

**Inference Stage**

Using the trained verification module $\mathcal{V}_\theta$, Alice determines if a test image $X_{\text{test}}$ is watermarked. For IID test data, conformal prediction provides FPR guarantees by setting $\tau$ as the empirical $\alpha$-quantile of scores:
$$
\mathbb{P}[g(X_{\text{test}}) = 1 \mid X_{\text{test}} \text{ is unwatermarked}] \leq \alpha.
$$

 





## Safety in RAG Systems
 
 
The rise of Retrieval-Augmented Generation (RAG) systems introduced in [Chapter 8](https://genai-course.jding.org/rag/index.html) has gained significant attention for its capabilities, particularly in applications like medical Q&A. However, safety considerations for RAG systems remain largely underexplored, leaving critical vulnerabilities unaddressed.

#### Universal Poisoning Attacks**

Recent research highlights [universal poisoning attacks](https://arxiv.org/pdf/2409.17275), where adversarial documents are injected into large-scale retrieval corpora (e.g., Wikipedia, PubMed) to ensure these documents rank highly for specific queries. By exploiting the reliance of Retrieval-Augmented Generation (RAG) systems on dense retrievers that map queries and documents into high-dimensional embedding spaces, these attacks demonstrate high success rates across 225 combinations of corpora, retrievers, and queries in medical Q&A.

Consequences:
- **Leakage of Personally Identifiable Information (PII)**: Poisoned documents can expose sensitive data tied to specific queries.
- **Adversarial Recommendations**: False or harmful medical advice can be injected and retrieved as trusted information.
- **Jailbreaking Language Models**: Poisoned retrievals can trigger models to bypass safety filters during inference.



**Figure: An illustration of universal poisoning attacks. The attacker can append a variety of adversarial information to a question to create a poisoned document and then inject it into the corpus. Upon querying the attacker-specified question, the poisoned document will be retrieved with a high ranking. These retrieved (poisoned) documents will lead to a variety of safety issues.** [source image](https://arxiv.org/pdf/2409.17275)
<div style="text-align:center; margin-bottom: 50px;">
    <img src="../_static/img/safety_rag.png" alt="Sample Aug" width="600" style="display:block; margin:auto;">
</div>



#### Vulnerability Mechanism: Orthogonal Augmentation Property

The study identifies an intriguing mechanism in dense retrievers that caused vulnerability: the **orthogonal augmentation property**, where concatenating adversarial content to a query shifts the poisoned document's embedding orthogonally to the original query. This orthogonal shift preserves high similarity between the poisoned document and the query, ensuring its high ranking during retrieval.

Additionally, clean retrieved documents are often loosely related to their queries, leaving a gap that attackers can exploit. For example, an average angle of around 70 degrees was observed between query embeddings and clean retrieved documents, indicating significant dissimilarity and increasing the effectiveness of poisoning attacks.

The attacks are shown to be robust even when attackers only have access to **paraphrased** versions of target queries. This enhances the practicality and severity of the threat, as precise query matches are not necessary for successful poisoning. This highlights a broader risk for retrieval systems in scenarios where query variability is expected.




<!-- TODO:
In the future, add data privacy and Interval Privacy. -->


## References

A Unified Framework for Inference-Stage Backdoor Defenses. [paper](https://openreview.net/pdf?id=4zWEyYGGfI)

Understanding Backdoor Attacks through the Adaptability Hypothesis. [paper](https://proceedings.mlr.press/v202/xian23a/xian23a.pdf)

Demystifying Poisoning Backdoor Attacks from a Statistical Perspective. [paper](https://openreview.net/pdf?id=BPHcEpGvF8)

AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models. [paper](https://arxiv.org/pdf/2401.09002v2)

RAW: A Robust and Agile Plug-and-Play Watermark Framework for AI-Generated Images with Provable Guarantees. [paper](https://openreview.net/pdf?id=ogaeChzbKu), [code](https://github.com/jeremyxianx/RAWatermark)

On the Vulnerability of Retrieval-Augmented Generation within Knowledge-Intensive Application Domains. [paper](https://arxiv.org/pdf/2409.17275)

PRADA: Protecting against DNN Model Stealing Attacks. [paper](https://arxiv.org/pdf/1805.02628)

Information Laundering for Model Privacy. [paper](https://openreview.net/pdf?id=dyaIRud1zXg)


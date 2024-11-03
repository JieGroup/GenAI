# 9. Safty in Generative AI
 

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
  
  

## References

[1] A Unified Framework for Inference-Stage Backdoor Defenses. [paper](https://openreview.net/pdf?id=4zWEyYGGfI)

[2] Understanding Backdoor Attacks through the Adaptability Hypothesis. [paper](https://proceedings.mlr.press/v202/xian23a/xian23a.pdf)

[3] Demystifying Poisoning Backdoor Attacks from a Statistical Perspective. [paper](https://openreview.net/pdf?id=BPHcEpGvF8)



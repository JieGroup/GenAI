
# 5. Diffusion Models

## Introduction

This tutorial discusses the essential ideas underlying diffusion models, which are the foundation of many modern generative tools. These models are particularly powerful in applications like text-to-image and text-to-video generation.

## Contents

1. [The Basics: Variational Auto-Encoder (VAE)](#vae)
    - [VAE Setting](#vae-setting)
    - [Evidence Lower Bound](#evidence-lower-bound)
    - [Training VAE](#training-vae)
    - [Loss Function](#loss-function)
    - [Inference with VAE](#inference-with-vae)
2. [Denoising Diffusion Probabilistic Model (DDPM)](#ddpm)
    - [Building Blocks](#building-blocks)
    - [The magical scalars √αt and 1-αt](#magical-scalars)
3. [Score-Matching Langevin Dynamics (SMLD)](#smld)
4. [Stochastic Differential Equation (SDE)](#sde)
5. [Conclusion](#conclusion)

## 1. The Basics: Variational Auto-Encoder (VAE) <a name="vae"></a>

### 1.1 VAE Setting <a name="vae-setting"></a>

A Variational Auto-Encoder (VAE) is a generative model that learns to generate images from a latent code. It consists of an encoder-decoder pair. The encoder converts the input image \(x\) into a latent vector \(z\), while the decoder reconstructs \(x\) from \(z\).

#### Example

Consider a random variable \(X\) distributed according to a Gaussian mixture model with a latent variable \(z \in \{1, \ldots, K\}\) denoting the cluster identity. The conditional distribution of \(X\) given \(Z\) is \(p_{X|Z}(x|k) = N(x|\mu_k, \sigma^2_kI)\).

### 1.2 Evidence Lower Bound <a name="evidence-lower-bound"></a>

The Evidence Lower Bound (ELBO) is a loss function used to optimize the VAE. It is defined as:

\[ \text{ELBO}(x) = \mathbb{E}_{q_\phi(z|x)} \left[ \log \frac{p(x, z)}{q_\phi(z|x)} \right] \]

ELBO is a lower bound for the prior distribution \(\log p(x)\), and maximizing ELBO helps in maximizing \(\log p(x)\).

### 1.3 Training VAE <a name="training-vae"></a>

To train a VAE, we use the ground truth pairs \((x, z)\). The encoder generates \(z\) from the distribution \(q_\phi(z|x)\), which is a Gaussian distribution with parameters \(\mu_\phi(x)\) and \(\sigma^2_\phi(x)\) estimated by neural networks.

### 1.4 Loss Function <a name="loss-function"></a>

The training loss of VAE is:

\[ \arg\max_{\phi, \theta} \left\{ \frac{1}{L} \sum_{\ell=1}^{L} \log p_\theta(x^{(\ell)}|z^{(\ell)}) - D_{KL}(q_\phi(z|x^{(\ell)}) \| p(z)) \right\} \]

### 1.5 Inference with VAE <a name="inference-with-vae"></a>

For inference, a latent vector \(z\) sampled from \(p(z) = N(0, I)\) is fed into the decoder to generate an image \(x\).

## 2. Denoising Diffusion Probabilistic Model (DDPM) <a name="ddpm"></a>

### 2.1 Building Blocks <a name="building-blocks"></a>

The DDPM model consists of a sequence of states \(x_0, x_1, \ldots, x_T\). The transition from \(x_{t-1}\) to \(x_t\) is realized by a denoiser.

#### Transition Distribution

The transition distribution \(q_\phi(x_t|x_{t-1})\) is defined as:

\[ q_\phi(x_t|x_{t-1}) = N(x_t | \sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)I) \]

### 2.2 The Magical Scalars √αt and 1-αt <a name="magical-scalars"></a>

The choice of the scalars \(\sqrt{\alpha_t}\) and \(1-\alpha_t\) ensures that the variance magnitude is preserved over iterations. 

## 3. Score-Matching Langevin Dynamics (SMLD) <a name="smld"></a>

Score-Matching Langevin Dynamics (SMLD) involves estimating the score function and using Langevin dynamics to sample from the data distribution.

## 4. Stochastic Differential Equation (SDE) <a name="sde"></a>

Stochastic Differential Equations (SDEs) describe the evolution of a system over time with a deterministic part and a stochastic part.

## 5. Conclusion <a name="conclusion"></a>

Diffusion models, including VAEs, DDPMs, SMLD, and SDEs, provide powerful frameworks for generative tasks. Understanding these models involves grasping the principles of variational inference, score matching, and stochastic processes.

---

This tutorial provides a foundation for understanding and applying diffusion models in various generative applications. For further reading and practical implementation, refer to the original research papers and online tutorials.

---

### References

- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.
- Other references as mentioned in the tutorial.

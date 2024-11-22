# 10. AI Safty
 

## Poisoning Backdoor Attacks
    

As machine learning models become integral to various applications, their safety emerges as a critical concern. In real-world scenarios, these models are often constructed from datasets whose intricacies may be obscured from users. This lack of transparency poses a risk for exploitation through **backdoor attacks**, a growing concern in AI security. These attacks are designed to make a model operate normally until it encounters specific, altered inputs that activate the backdoor, causing the model to behave unpredictably, as demonstrated in the following figure.

  
**Figure: An example of a backdoor attack that compromises the traffic sign classifier for autonomous driving.**
<div style="text-align:center; margin-bottom: 50px;">
    <img src="../_static/img/fig_stopsign.png" alt="Sample Aug" width="300" style="display:block; margin:auto;">
</div>
  



**Threat Model**

- **Data Poisoning**: Adversaries inject crafted data into the training set
- **Trigger Activation**: The backdoor is activated only when the specific trigger pattern appears in the input
- **Stealthiness**: The attack remains undetected during model training and validation, as the model behaves normally on untriggered inputs.
- **Attack Goals**: A powerful backdoor attack has a **dual-goal**: being stealthy and useful, meaning that it 
  - prompt the compromised model to exhibit manipulated behavior when a specific attacker-defined trigger is present, and 
  - maintain normal functionality in the absence of the trigger, rendering the attack difficult to detect.


  
### Demystifying Poisoning Backdoor Attacks

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

- The **optimal trigger** direction is where the clean data distribution decays the most

- Constructing a **human-imperceptible backdoor attack could be more feasible** when the clean data distribution degenerates more


The above fundamental understanding also serves as a basis for developing improved defense mechanisms against backdoor attacks.
  


Below, we show how to backdoor a conditional diffusion model by training it on the backdoored MNIST dataset. We first define some basic components of diffusion models.


```python
''' 
This script mainly adapts the code from the following sources:
https://github.com/TeaPearce/Conditional_Diffusion_MNIST
'''
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import random

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=11):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), 
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w = 0.0):
        # We follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'. To make the fwd passes efficient, we concat two versions of the dataset, one with context_mask=0 and the other context_mask=1. We then mix the outputs with the guidance scale, w, where w>0 means more guidance
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,11).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
```

Next, we create the backdoored MNIST dataset. Please try to plot the backdoored samples and see if you can spot the backdoor trigger.

```python
class ExtendedMNISTDataset(Dataset):
    def __init__(self, original_dataset, num_additional_samples_ratio = 0.1):
        self.original_dataset = original_dataset
        self.num_additional_samples = int(len(original_dataset) * num_additional_samples_ratio)
        self.additional_samples = self._generate_additional_samples()

    def _generate_additional_samples(self):
        additional_samples = []
        for _ in range(self.num_additional_samples):
            while True:
                index = random.randint(0, len(self.original_dataset) - 1)
                data, label = self.original_dataset[index]
                if label == 7:
                    break
            modified_data = data.clone()
            modified_data[0, -2:, -2:] = 1.0
            modified_label = 10
            additional_samples.append((modified_data, modified_label))
        return additional_samples

    def __len__(self):
        return len(self.original_dataset) + self.num_additional_samples

    def __getitem__(self, index):
        if index < len(self.original_dataset):
            data, label = self.original_dataset[index]
        else:
            data, label = self.additional_samples[index - len(self.original_dataset)]
        return data, label
```
Then, we train the backdoored model on the backdoored MNIST dataset.

```python
def train_mnist():

    # hardcoding these here
    n_epoch = 20
    batch_size = 256
    n_T = 400 # 500
    device = "cuda:0"
    n_classes = 11
    n_feat = 128 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = False
    ############ change this to your save directory #############
    save_dir = '/path/to/save/dir/'
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

    dataset = MNIST("./data", train=True, download=True, transform=tf)
    
    extended_train_dataset = ExtendedMNISTDataset(dataset, num_additional_samples_ratio=0.1)

    # Create a DataLoader for your extended dataset
    dataloader = DataLoader(extended_train_dataset, batch_size=batch_size, shuffle=True, num_workers=14, pin_memory=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4*n_classes
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample/n_classes)):
                        try: 
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k+(j*n_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

                if ep%5 == 0 or ep == int(n_epoch-1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample/n_classes)):
                            for col in range(n_classes):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                                plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                        return plots
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                    ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
        # optionally save model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")
            
train_mnist()
```
  

### Mitigating Backdoor Attack
    
  

The hidden nature of backdoor attacks underscores the urgency of developing robust defenses. Current strategies fall into two categories: **inference-stage defenses** for detecting backdoored data at the point of use, and **training-stage defenses** to prevent neural networks from learning from backdoored training data. 

As Machine Learning as a Service (MLaaS) becomes increasingly common, there is a growing need for real-time, on-the-fly defense mechanisms against backdoor attacks. In such a context, inference-stage defenses are critical because they offer the last line of defense, operating at the point where the model is actually used to make predictions on new data. However, they often lack theoretical foundation and are typically limited to vision tasks, leaving a gap in natural language processing applications.


**Figure: Illustration of inference-stage backdoor defenses. The backdoor triggers in text queries are indicated in red, while for the image queries, the backdoor triggers consist of a square patch located in the lower-right corner for the traffic sign and a hello kitty embedding added to an image of a dog.**
<div style="text-align:center; margin-bottom: 50px;">
    <img src="../_static/img/fig_backdoor_defenses.png" alt="Sample Aug" width="650" style="display:block; margin:auto;">
</div>

 

A recent framework is known as Conformal Backdoor Detection (CBD). This framework is tailored to combat backdoor attacks by pinpointing query inputs that have been tampered with by adversaries. CBD establishes a benchmark that achieves the state-of-the-art backdoor detection accuracy in the field. 
  

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

The defender aims to devise a detector $g$ to

$$
\text{maximize } \quad \mathbb{P}\bigl\{ g(X ; \tau) = 1 \mid X \text{ is backdoor} \bigr\}, \\ 
\text{while controlling the false positive rate (FPR):} \quad 
\mathbb{P}\bigl\{ g(X ; \tau) = 1 \mid X \text{ is clean} \bigr\}.
$$

To effectively mitigate backdoor attacks on Deep Neural Networks, CBD employs a conformal prediction framework to precisely control the False Positive Rate (FPR). By leveraging a decision threshold based on empirical data distribution, this approach remains statistically justified without depending on strong distributional assumptions.
  

### Jailbreak Attacks

Jailbreak attacks represent a form of inference-stage vulnerability in generative AI models, analogous to backdoor attacks at the training stage. These attacks manipulate an AI system to bypass its safety constraints, generating unintended or harmful outputs when given crafted prompts.


**Figure: While backdoor attacks require tampered training data, jailbreak attacks operate during inference and target vulnerabilities in the model's prompt handling. [image source](https://arxiv.org/pdf/2401.09002v2)**
<div style="text-align:center; margin-bottom: 50px;">
    <img src="../_static/img/safety_jailbreak.png" alt="Sample Aug" width="300" style="display:block; margin:auto;">
</div>


Despite their increasing prevalence, understanding the mechanisms behind jailbreak attacks remains limited, and their construction is largely ad hoc. 




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

**Definition**:  A learned model is a kernel $p: \mathcal{X} \times \mathcal{Y} \rightarrow [0,1]$, which induces a class of conditional distributions $\{p(\cdot \mid x): x \in \mathcal{X}\}$.

**Definition**:  An information-laundered model with respect to a given model $K_*$ is a model $K$ that consists of three internal kernels:

$$
K = K_1 \circ K_* \circ K_2,
$$

where $K_1$ and $K_2$ are input and output perturbation kernels, respectively. This is illustrated in the figure below.



**Figure: Illustration of Information Laundering (a) Alice's effective system for public use, and (b) Alice's idealistic system not for public use.** [source image](https://openreview.net/pdf?id=dyaIRud1zXg)
<div style="text-align:center; margin-bottom: 50px;">
    <img src="../_static/img/safety_laundering.png" alt="Sample Aug" width="600" style="display:block; margin:auto;">
</div>

Denote the kernels representing the authentic model, input kernel, output kernel, and the information-laundered model as $p(\cdot \mid \cdot), p_{K_1}(\cdot \mid \cdot), p_{K_2}(\cdot \mid \cdot), p_{K_*}(\cdot \mid \cdot)$, respectively.

The information laundering objective minimizes the following function:

$$
L(p_{K_1}, p_{K_2}) =
\mathbb{E}_{X \sim p_X} \mathcal{D}(p(\cdot \mid X), p_{K_*}(\cdot \mid X)) + \beta_1 \mathcal{I}(X; \tilde{X}) + \beta_2 \mathcal{I}(Y; \tilde{Y}),
$$

where:
- $\mathcal{D}$ represents the divergence between the true model and the laundered model.
- $\mathcal{I}$ represents mutual information to control privacy leakage between original and perturbed variables.

Using the calculus of variations, the framework derives closed-form optimal perturbation kernels for both inputs and outputs, enabling practical implementations of privacy-utility tradeoffs.





## Watermarking 

The rapid advancements in Generative AI, particularly in diffusion models (DM) like Stable Diffusion and DALLE-2, have significantly improved image generation quality. However, these innovations raise concerns, including **DeepFake** misuse and **copyright infringement**. To mitigate misuse, **watermarking** techniques offer a way to identify machine-generated content by embedding distinct signals.

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

#### Universal Poisoning Attacks

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


# 4. Human-value Alignment
<!-- and Human Value Alignment -->


## (Recall) Overview of Using LLM in Practice

```{mermaid}
graph LR
    UseCase[Define Science Domain] --> DataCuration[Data Curation]
    DataCuration --> Architecture[Set Architecture]
    Architecture --> Training[Training]
    Training --> Base[Base Model]
```
```{mermaid}
graph LR
    BaseModel[Base Model] --> FineTuning[Supervised Fine-tuning]
	UseCase[Define Application Case] --> FineTuning[Supervised Fine-tuning]
    FineTuning --> RLHF[Human Value Alignment]
```

We will go through human value alignment problems in this chapter.



# Reference


- Supervised Fine-tuning Trainer. [code](https://huggingface.co/docs/trl/main/en/sft_trainer)

- Prefix-Tuning: Optimizing Continuous Prompts for Generation. [paper](https://arxiv.org/pdf/2101.00190)

- LoRA: Low-Rank Adaptation of Large Language Models. [paper](https://arxiv.org/pdf/2106.09685)

- ColA: Collaborative Adaptation with Gradient Learning. [paper](https://arxiv.org/pdf/2404.13844)

- List of Large Language Model applied to Tabular Data. [papers](https://github.com/johnnyhwu/Awesome-LLM-Tabular)

- Drift to Remember. [paper](https://arxiv.org/abs/2409.13997)

This lecture note includes code examples and concepts adapted from the following sources. We acknowledge and thank the authors for their contributions to the open-source community.

- Stanford Alpaca: An Instruction-following LLaMA Model. [code](https://github.com/tatsu-lab/stanford_alpaca)

- Self-Instruct: Aligning LM with Self Generated Instructions
. [paper](https://arxiv.org/abs/2212.10560), [code](https://github.com/yizhongw/self-instruct)

- 52K instruction-following data. [data](https://huggingface.co/datasets/yahma/alpaca-cleaned)
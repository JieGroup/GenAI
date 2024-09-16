# 4. Finetuning and Human Value Alignment




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


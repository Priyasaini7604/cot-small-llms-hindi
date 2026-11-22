# Chain-of-Thought in Small Open LLMs (GSM8K, English vs Hindi)

Does chain-of-thought (CoT) prompting help small (0.5B-7B) open models on grade-school math, and does the benefit hold in Hindi?

## Setup
- **Dataset:** first 200 questions of the GSM8K test split (`openai/gsm8k`), plus a Hindi translation of the same 200.
- **Models:** Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B.
- **Strategies:** zero-shot, few-shot (4 examples), CoT ("Let's think step by step"), self-consistency (5 samples, temperature 0.7, majority vote).
- Instruction text is always English; only the question changes language.
- Team-wide settings: seed 42, max_new_tokens 512, greedy decoding (except self-consistency), 200 questions.

## Repository structure
```
code/
  pipeline.ipynb            run experiments (one member = one model)
  translation.ipynb  English -> Hindi translation helper + merge
  Analysis.ipynb
data/                       gsm8k_200_en.json, gsm8k_200_hi.json, fewshot_en.json, fewshot_hi.json
result/                    <model_tag>_<strategy>_<lang>.json  (8 files per model, 32 total)
analysis/                   results table and graphs
report/                     final report and slides
```

## How to run
1. Open `code/pipeline.ipynb` in Google Colab with a GPU runtime.
2. Put the shared data folder in your Drive (`DL_CoT_Project/data/`).
3. In Cell 1 change only `MODEL`, `MODEL_TAG`, `BS`, `USE_4BIT`.
4. Run Cells 1-6, then Cell 7 (quick test), Cell 8 (full runs), Cell 9 (summary).

Each run saves `results/<model_tag>_<strategy>_<lang>.json` with the accuracy and every model output.

## Results
_To be filled in after all runs are complete._

## Team
| Member | Model | Contribution |
|---|---|---|
| Priya | Qwen2.5-0.5B | pipeline, data preparation|
| Tamnna Praveen | Qwen2.5-1.5B | Hindi translation (Q1-67), Method |
| Rohit kumar | Qwen2.5-3B | Hindi translation (Q68-134), results table and graphs |
| Prerna | Qwen2.5-7B | Hindi translation (Q135-200),merge |
|All Members| analysis|



## References
1. Wei et al., Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, NeurIPS 2022. https://arxiv.org/abs/2201.11903
2. Cobbe et al., Training Verifiers to Solve Math Word Problems (GSM8K), 2021. https://arxiv.org/abs/2110.14168

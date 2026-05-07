# results

Evaluation results of multiple LLM personalization methods on the YNTP-100 benchmark. Each subfolder corresponds to one method and contains model responses and computed scores across the three languages (Chinese, English, Japanese).

## Structure

```
results/
├── Aligner/
│   ├── response/
│   └── score/
├── Chain-of-Thought/
│   ├── response/
│   └── score/
├── DPO/
│   ├── response/
│   └── score/
├── Fine-tuning/
│   ├── response/
│   └── score/
├── Prompt Eng. (few-shot)/
│   ├── response/
│   └── score/
└── Prompt Eng. (zero-shot)/
    ├── response/
    └── score/
```

Each method folder contains two subfolders:

- **`response/`** — Raw model-generated responses, organized by model (e.g., `Qwen_Qwen3-14B/`, `elyza_Llama-3-ELYZA-JP-8B/`, `meta-llama_Llama-3.1-8B-Instruct/`).
- **`score/`** — Evaluation scores for each model's responses, organized in the same model-level subfolders.

## Methods

| Method | Description |
|--------|-------------|
| `Aligner` | Post-hoc alignment via the Aligner framework |
| `Chain-of-Thought` | Reasoning-prompted generation (CoT) |
| `DPO` | Direct Preference Optimization fine-tuning |
| `Fine-tuning` | Supervised fine-tuning on preference data |
| `Prompt Eng. (few-shot)` | Few-shot prompting with example demonstrations |
| `Prompt Eng. (zero-shot)` | Zero-shot prompting without demonstrations |


## Usage

Response files correspond to the 100 anonymized conversation samples in `anonymous_100/` (across `cn/`, `en/`, `jp/`). Score files contain per-sample evaluation metrics (BERTScore, BLEURT, etc.) computed against the ground-truth user responses.


# AutoInsight-R: Self-Healing GenAI for Reliable Data Analysis

A self-healing data analysis pipeline powered by Claude. AutoInsight-R accepts a natural language request, generates Python code, executes it, and automatically repairs failures — without any human intervention.

## How it works

1. You describe an analysis task in plain English.
2. Claude (Haiku by default) generates Python code with your dataset's schema pre-injected.
3. The code runs in an isolated subprocess — real pandas, matplotlib, sklearn.
4. If it fails, the error is diagnosed, targeted context is injected, and Claude repairs the code.
5. On the final retry, the pipeline automatically escalates to Sonnet as a hard fallback.

### Model selection strategy

| Situation | Model used |
|---|---|
| Standard EDA task | Haiku (fast, cheap) |
| ML keywords detected in prompt | Sonnet from the start |
| Any task on final retry | Sonnet (hard fallback) |

## Experimental Results

Evaluated on 30 tasks adapted from [DS-1000](https://github.com/xlang-ai/DS-1000), covering pandas, numpy, visualization, and modeling.

### Self-Healing vs Baseline

| Model | Baseline (one-shot) | Self-Healing | Improvement |
|-------|---------------------|--------------|-------------|
| Claude Sonnet | 86.7% | 96.7% | +10% |
| Claude Haiku | 86.7% | 100% | +13.3% |

### Ablation Study — Effect of max_retries

| max_retries | Sonnet | Haiku |
|-------------|--------|-------|
| 1 | 86.7% | 86.7% |
| 2 | 96.7% | 100% |
| 3 | 96.7% | 100% |

### Success Rate by Difficulty

| Difficulty | Baseline | Self-Healing |
|------------|----------|--------------|
| Medium | 95.5% | 100% |
| Hard | 62.5% | 87.5% |

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Project Structure

```
AutoInsight-R/
├── app.py                  # Streamlit UI
├── pipeline.py             # Self-healing pipeline logic
├── code_executor.py        # Subprocess sandbox execution
├── model_selector.py       # Keyword-based model heuristic
├── requirements.txt
├── autoinsight.ipynb       # Benchmark experiments (Google Colab)
├── datasets/
│   ├── Iris.csv
│   ├── titanic.csv
│   └── housing.csv
└── benchmarks/
    └── prompts.json        # 30 DS-1000 style benchmark tasks
```

## Notes

- Your API key is entered in the sidebar and used only for the current session.
- Generated code runs in a temporary directory that is deleted after each run.
- Charts are captured via plt.show() and displayed inline in the UI.
- Supported built-in datasets: Iris, Titanic, Housing (loaded from GitHub).
- You can also upload any CSV file.

## Team

- Yuhan Yan (yy3630)
- Yunjie Huang (yh3976)
- Jade Chang (jc6616)

Columbia University — Spring 2025

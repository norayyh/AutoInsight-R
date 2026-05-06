# AutoInsight

A self-healing data analysis pipeline powered by Claude.

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

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Project structure

```
autoinsight/
├── app.py              # Streamlit UI
├── pipeline.py         # Self-healing pipeline logic
├── code_executor.py    # Subprocess sandbox execution
├── model_selector.py   # Keyword-based model heuristic
└── requirements.txt
```

## Notes

- Your API key is entered in the sidebar and used only for the current session.
- Generated code runs in a temporary directory that is deleted after each run.
- Charts are captured via plt.show() and displayed inline in the UI.
- Supported built-in datasets: Iris, Titanic, Housing (loaded from GitHub).
- You can also upload any CSV file.

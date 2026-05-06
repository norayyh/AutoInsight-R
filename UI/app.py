"""
app.py
------
AutoInsight — Streamlit UI for the self-healing data analysis pipeline.

Layout:
    Sidebar  : API key, dataset selector (built-in or upload), schema preview.
    Main     : Task input, live pipeline step cards, output (stdout + charts).

Pipeline model strategy (visible to user):
    - Haiku  : default for straightforward EDA tasks.
    - Sonnet : auto-selected upfront for complex tasks (ML keywords detected),
               and always used as a hard fallback on the final repair attempt.
"""

import io
import os
import tempfile

import pandas as pd
import streamlit as st

from model_selector import model_label, MODEL_HAIKU, MODEL_SONNET, MAX_RETRIES
from pipeline import run_pipeline, StepResult, PipelineResult

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "AutoInsight",
    page_icon  = "🔍",
    layout     = "wide",
)

# ── built-in datasets ─────────────────────────────────────────────────────────

BUILTIN_URLS = {
    "Iris":     "https://raw.githubusercontent.com/norayyh/AutoInsight-R/main/datasets/Iris.csv",
    "Titanic":  "https://raw.githubusercontent.com/norayyh/AutoInsight-R/main/datasets/titanic.csv",
    "Housing":  "https://raw.githubusercontent.com/norayyh/AutoInsight-R/main/datasets/housing.csv",
}

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_builtin(url: str) -> bytes:
    import urllib.request
    with urllib.request.urlopen(url) as r:
        return r.read()


def build_schema(df: pd.DataFrame) -> dict:
    sample_str = df.head(5).to_string(index=False)
    return {
        "columns":    df.columns.tolist(),
        "dtypes":     {c: str(t) for c, t in df.dtypes.items()},
        "shape":      df.shape,
        "sample_str": sample_str,
    }


def save_csv_to_temp(raw_bytes: bytes) -> str:
    """Write CSV bytes to a named temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(raw_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def status_icon(status: str) -> str:
    return {"running": "⏳", "done": "✅", "success": "✅", "fail": "❌"}.get(status, "•")


def model_chip(model: str) -> str:
    label = model_label(model)
    color = "#0F6E56" if "haiku" in model else "#185FA5"
    bg    = "#E1F5EE"  if "haiku" in model else "#E6F1FB"
    return (
        f'<span style="background:{bg};color:{color};'
        f'padding:2px 8px;border-radius:999px;font-size:12px;'
        f'font-weight:500;">{label}</span>'
    )


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("AutoInsight")
    st.caption("Self-healing data analysis pipeline")

    st.divider()

    # API key
    api_key = st.text_input(
        "Anthropic API key",
        type     = "password",
        help     = "Your key is used only for this session and never stored.",
        placeholder = "sk-ant-...",
    )

    st.divider()

    # Dataset selection
    st.subheader("Dataset")
    source = st.radio("Source", ["Built-in", "Upload"], horizontal=True, label_visibility="collapsed")

    raw_csv_bytes: bytes | None = None
    dataset_name  = ""

    if source == "Built-in":
        choice = st.selectbox("Select dataset", list(BUILTIN_URLS.keys()))
        if st.button("Load", use_container_width=True):
            with st.spinner(f"Fetching {choice}..."):
                raw_csv_bytes = fetch_builtin(BUILTIN_URLS[choice])
                st.session_state["csv_bytes"] = raw_csv_bytes
                st.session_state["dataset_name"] = choice
                st.success(f"{choice} loaded.")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            raw_csv_bytes = uploaded.read()
            st.session_state["csv_bytes"] = raw_csv_bytes
            st.session_state["dataset_name"] = uploaded.name

    # Schema preview
    if "csv_bytes" in st.session_state:
        df_preview = pd.read_csv(io.BytesIO(st.session_state["csv_bytes"]))
        schema     = build_schema(df_preview)

        st.divider()
        st.subheader("Schema")
        st.caption(f"{schema['shape'][0]:,} rows · {schema['shape'][1]} columns")

        schema_df = pd.DataFrame({
            "Column": schema["columns"],
            "Type":   [schema["dtypes"][c] for c in schema["columns"]],
        })
        st.dataframe(schema_df, hide_index=True, use_container_width=True, height=220)

    st.divider()

    # Model strategy legend
    st.subheader("Model strategy")
    st.markdown(
        f"**Default** — {model_label(MODEL_HAIKU)} for standard EDA  \n"
        f"**Auto-upgrade** — {model_label(MODEL_SONNET)} when ML keywords detected  \n"
        f"**Hard fallback** — {model_label(MODEL_SONNET)} on final retry (attempt {MAX_RETRIES})"
    )


# ── main area ─────────────────────────────────────────────────────────────────

st.header("Analysis task")

prompt = st.text_area(
    "Describe what you want to analyse",
    placeholder = "e.g. Plot a histogram of each numeric column and print summary statistics.",
    height      = 90,
    label_visibility = "collapsed",
)

run_disabled = not api_key or "csv_bytes" not in st.session_state or not prompt.strip()
run_clicked  = st.button(
    "Run pipeline",
    type             = "primary",
    disabled         = run_disabled,
    use_container_width = True,
)

if not api_key:
    st.info("Enter your Anthropic API key in the sidebar to get started.")
elif "csv_bytes" not in st.session_state:
    st.info("Load or upload a dataset using the sidebar.")

# ── pipeline execution ────────────────────────────────────────────────────────

if run_clicked:
    csv_bytes = st.session_state["csv_bytes"]
    df_run    = pd.read_csv(io.BytesIO(csv_bytes))
    schema    = build_schema(df_run)

    # Write CSV to a temp file so the subprocess executor can read it.
    csv_path  = save_csv_to_temp(csv_bytes)

    st.divider()
    st.subheader("Pipeline")

    # Placeholder containers for each step — updated in place as steps arrive.
    step_slots: dict = {}
    final_result: PipelineResult | None = None

    gen = run_pipeline(
        prompt   = prompt,
        schema   = schema,
        csv_path = csv_path,
        api_key  = api_key,
    )

    try:
        while True:
            step: StepResult = next(gen)

            slot_key = f"{step.step}_{step.attempt}"
            if slot_key not in step_slots:
                step_slots[slot_key] = st.empty()

            slot = step_slots[slot_key]

            # Build step card content
            with slot.container():
                icon  = status_icon(step.status)
                label_map = {"generate": "Generate code", "execute": "Execute", "repair": "Diagnose & repair"}
                label = label_map.get(step.step, step.step)

                cols = st.columns([0.04, 0.96])
                with cols[0]:
                    st.markdown(f"### {icon}")
                with cols[1]:
                    header_parts = [f"**{label}** — attempt {step.attempt}"]
                    if step.model:
                        header_parts.append(model_chip(step.model))
                    if step.note:
                        header_parts.append(f"<span style='color:gray;font-size:12px;'>{step.note}</span>")
                    st.markdown(" &nbsp; ".join(header_parts), unsafe_allow_html=True)

                    if step.status == "running":
                        st.caption("Working...")

                    if step.code and step.status in ("done", "success"):
                        with st.expander("View generated code", expanded=False):
                            st.code(step.code, language="python")

                    if step.stderr and step.status == "fail":
                        with st.expander("Error details", expanded=True):
                            st.code(step.stderr, language="text")

                    if step.stdout and step.status == "success":
                        if step.stdout.strip():
                            with st.expander("Printed output", expanded=True):
                                st.text(step.stdout)

    except StopIteration as e:
        final_result = e.value
    finally:
        # Clean up the temp CSV file.
        try:
            os.unlink(csv_path)
        except OSError:
            pass

    # ── output section ────────────────────────────────────────────────────────

    st.divider()
    st.subheader("Result")

    if final_result is None:
        st.error("Pipeline did not return a result. Check your API key and try again.")
    elif final_result.success:
        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Status",       "Success ✅")
        m2.metric("Attempts used", f"{final_result.attempts} / {MAX_RETRIES}")
        m3.metric("Retries",       final_result.attempts - 1)

        # Printed output
        if final_result.stdout:
            st.markdown("**Output**")
            st.text(final_result.stdout)

        # Charts
        if final_result.images:
            st.markdown("**Charts**")
            cols = st.columns(min(len(final_result.images), 2))
            for i, img_bytes in enumerate(final_result.images):
                cols[i % 2].image(img_bytes, use_container_width=True)

        # Final code
        with st.expander("Final code", expanded=False):
            st.code(final_result.code, language="python")

    else:
        m1, m2 = st.columns(2)
        m1.metric("Status",  "Failed ❌")
        m2.metric("Attempts", f"{final_result.attempts} / {MAX_RETRIES}")

        st.error("All attempts exhausted. See the error below.")
        st.code(final_result.stderr or "No error captured.", language="text")

        with st.expander("Last attempted code", expanded=False):
            st.code(final_result.code, language="python")

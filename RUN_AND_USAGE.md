# Run and Usage Guide

This document explains exactly how to run the notebook pipeline and chatbot.

## 1) Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Confirm Required Files
Make sure these exist in the project root:
- `conversations.csv`
- `solution.ipynb`
- `rag_persona_system.py`
- `app.py`

## 3) Run Notebook Pipeline

### Option A: Jupyter
```bash
jupyter notebook solution.ipynb
```
Run cells top-to-bottom.

### Option B: VS Code / Cursor notebook runner
- Open `solution.ipynb`.
- Run all cells in order.

Expected generated files:
- `artifacts/messages.parquet`
- `artifacts/messages.csv`
- `artifacts/topic_checkpoints.json`
- `artifacts/message_checkpoints_100.json`
- `artifacts/message_chunks.json`
- `artifacts/persona.json`

## 4) Run Chatbot Locally
```bash
streamlit run app.py
```

Open local URL shown by Streamlit (usually `http://localhost:8501`).

## 5) Test Required Prompts
Inside chatbot, test:
- `What kind of person is this user?`
- `What are their habits?`
- `How do they talk?`

Verify:
- Answer text appears.
- Retrieved checkpoint summaries are shown.
- Retrieved message chunks are shown.
- Persona snapshot is visible.

## 6) Rebuild Artifacts from Scratch
```bash
rm -rf artifacts
```
Then rerun notebook from the first cell.

## 7) Cloud Deployment (Streamlit Community Cloud)
1. Push repo to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Create app using:
   - repo: your repository
   - branch: main
   - file: `app.py`
4. Add deployed URL to `README.md`.

## 8) Troubleshooting
- If `pyarrow` missing: `pip install pyarrow`.
- If notebook kernel mismatch: select the `.venv` Python interpreter.
- If app startup is slow first time: indexing is built at startup if artifacts are missing.
- If memory is tight: reduce `max_features` in vectorizers in `rag_persona_system.py`.

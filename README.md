# StudySmart — Group 139

Simple ML project: predict student pass/fail from 4 features.

```bash
pip install -r requirements.txt
python setup.py
uvicorn src.api:app --reload --port 8000
```

**Streamlit UI** (optional web form):

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501

| File | Description |
|------|-------------|
| `Group_139.ipynb` | Assignment notebook |
| `139.docx` | **Submission document** — upload to Taxila portal |
| `generate_docx.py` | Regenerate `139.docx` after edits |
| `src/pipeline.py` | Pipeline pattern |
| `src/api.py` | Microservices pattern |
| `streamlit_app.py` | Streamlit web UI (client) |

Regenerate data and model before running the app:

```bash
python setup.py
```

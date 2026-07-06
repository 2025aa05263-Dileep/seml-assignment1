"""StudySmart web UI — Streamlit client for PassPredictor."""

from pathlib import Path

import streamlit as st

from src.predict import PassPredictor

ROOT = Path(__file__).parent
MODEL = ROOT / "models" / "pass_model.joblib"


@st.cache_resource
def load_predictor() -> PassPredictor:
    if not MODEL.exists():
        raise FileNotFoundError("Model missing. Run: python setup.py")
    return PassPredictor(MODEL)


def main():
    st.set_page_config(page_title="StudySmart", page_icon="🎓", layout="centered")
    st.title("StudySmart")
    st.caption("Student pass/fail prediction — Group 139")

    try:
        predictor = load_predictor()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("pip install -r requirements.txt\npython setup.py\nstreamlit run streamlit_app.py")
        return

    with st.sidebar:
        st.subheader("Model metrics")
        if predictor.metrics:
            for key, value in predictor.metrics.items():
                st.metric(key.replace("_", " ").title(), value)
        else:
            st.info("No metrics stored in model artifact.")

    st.subheader("Student profile")
    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider("Study hours / week", 0, 60, 10)
        assignments_done = st.slider("Assignments completed", 0, 10, 5)
    with col2:
        attendance_percent = st.slider("Attendance %", 0, 100, 75)
        previous_score = st.slider("Previous exam score", 0, 100, 50)

    if st.button("Predict", type="primary", use_container_width=True):
        result = predictor.predict(
            study_hours=study_hours,
            attendance_percent=attendance_percent,
            assignments_done=assignments_done,
            previous_score=previous_score,
        )

        st.divider()
        if result["passed"]:
            st.success(f"**Likely Pass** — probability: {result['pass_probability']:.0%}")
        else:
            st.warning(f"**At Risk** — probability: {result['pass_probability']:.0%}")

        st.json(result)


if __name__ == "__main__":
    main()

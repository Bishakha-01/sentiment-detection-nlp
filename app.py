import re
import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="MoodLens | Sentiment Detection",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f7f8fc; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1200px; }
.hero {
    background: linear-gradient(120deg, #252c55 0%, #5147a5 58%, #8c78dc 100%);
    padding: 2rem 2.2rem; border-radius: 24px; color: white; margin-bottom: 1.2rem;
}
.hero h1 { font-family: 'Manrope', sans-serif; font-size: 2.3rem; margin: 0 0 .4rem 0; }
.hero p { color: #e5e7ff; margin: 0; font-size: 1.02rem; }
.eyebrow { text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 700; color: #bfc5ff; }
.metric-card {
    background: white; border: 1px solid #e8eaf4; border-radius: 18px;
    padding: 1.15rem 1.25rem; min-height: 120px;
}
.metric-label { color: #72768b; font-size: .86rem; margin-bottom: .45rem; }
.metric-value { color: #25294b; font-family: 'Manrope', sans-serif; font-size: 1.7rem; font-weight: 800; }
.metric-note { color: #8b8fa3; font-size: .78rem; margin-top: .3rem; }
.section-title { font-family: 'Manrope', sans-serif; font-weight: 800; color: #25294b; font-size: 1.25rem; margin: .5rem 0 .7rem; }
.result-card {
    background: white; border: 1px solid #e8eaf4; border-radius: 22px;
    padding: 1.5rem; text-align: center;
}
.small-muted { color: #777d94; font-size: .88rem; }
div[data-testid="stSidebar"] { background: #f0f1fa; }
div.stButton > button[kind="primary"] {
    background: #5147a5; border: 0; border-radius: 12px; font-weight: 700;
    padding: .65rem 1.2rem;
}
div.stButton > button[kind="primary"]:hover { background: #3e367f; border: 0; }
</style>
""", unsafe_allow_html=True)

EMOTION_META = {
    "joy":      {"emoji": "😊", "color": "#eab308", "description": "Positive, happy, or excited feelings"},
    "sadness":  {"emoji": "🌧️", "color": "#6488d8", "description": "Low, disappointed, or sorrowful feelings"},
    "anger":    {"emoji": "😠", "color": "#e46b65", "description": "Frustration, irritation, or anger"},
    "fear":     {"emoji": "😟", "color": "#9a7bd1", "description": "Worry, nervousness, or fear"},
    "love":     {"emoji": "💗", "color": "#e779ad", "description": "Affection, care, or warmth"},
    "surprise": {"emoji": "😲", "color": "#53b6ad", "description": "Unexpected or astonishing feelings"},
}

DATA_PATH = Path(__file__).resolve().parent / "train.txt"

def clean_text(text):
    """Lightweight cleaning. Negation words are retained because they affect sentiment."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_data(show_spinner=False)
def load_dataset(path_string, modified_time):
    path = Path(path_string)
    # The dataset format is: sentence;emotion
    data = pd.read_csv(path, sep=";", header=None, names=["text", "emotion"], engine="python")
    data = data.dropna(subset=["text", "emotion"])
    data["text"] = data["text"].astype(str).str.strip()
    data["emotion"] = data["emotion"].astype(str).str.strip().str.lower()
    data = data[(data["text"] != "") & (data["emotion"] != "")]
    return data.reset_index(drop=True)

@st.cache_resource(show_spinner="Training the sentiment model…")
def train_model(path_string, modified_time):
    data = load_dataset(path_string, modified_time)
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["emotion"], test_size=0.2, random_state=42, stratify=data["emotion"]
    )
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_features=80000,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1200, class_weight=None)
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    labels = sorted(data["emotion"].unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)
    # Refit on the full dataset for the interactive predictor.
    all_vec = vectorizer.fit_transform(data["text"])
    model.fit(all_vec, data["emotion"])
    return data, vectorizer, model, accuracy, report, labels, cm

if not DATA_PATH.exists():
    st.error("I couldn't find `train.txt`. Put it in the same folder as `app.py`, then run the app again.")
    st.stop()

modified_time = DATA_PATH.stat().st_mtime
try:
    df, vectorizer, model, test_accuracy, report, labels, cm = train_model(str(DATA_PATH), modified_time)
except Exception as exc:
    st.error(f"Could not load the dataset or train the model: {exc}")
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 💬 MoodLens")
    st.caption("Text emotion detection")
    page = st.radio(
        "NAVIGATION",
        ["🏠 Overview", "🔎 Analyze text", "📊 Dataset insights", "🧠 Model details"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**Model status**")
    st.success("Model trained")
    st.caption(f"{len(df):,} labelled examples · {len(labels)} emotion classes")
    st.caption("Predictions are estimates, not a definitive reading of someone's feelings.")

# ---------- Overview ----------
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">NLP · MACHINE LEARNING</div>
      <h1>Understand the emotion behind the words.</h1>
      <p>Explore a text emotion classifier trained on your dataset. Paste a sentence and see the predicted emotion and class probabilities.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, label, value, note in [
        (c1, "Training examples", f"{len(df):,}", "Rows loaded from train.txt"),
        (c2, "Emotion classes", str(len(labels)), "Labels available in the dataset"),
        (c3, "Test accuracy", f"{test_accuracy:.1%}", "Held-out 20% split, before final refit"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Try a quick example</div>", unsafe_allow_html=True)
    examples = {
        "Joy": "I finally finished my project and I am so proud of myself!",
        "Sadness": "I feel lonely and I miss the people I used to talk to.",
        "Anger": "I am frustrated because nothing is working the way I expected.",
        "Fear": "I am nervous about what might happen tomorrow.",
        "Love": "I feel lucky to have people who care about me.",
        "Surprise": "I never expected that news. I am completely amazed!",
    }
    chosen = st.selectbox("Choose a sample sentence", list(examples.keys()))
    if st.button("Analyze this example", type="primary"):
        st.session_state["text_to_analyze"] = examples[chosen]
        st.session_state["run_analysis"] = True
        st.rerun()
    st.info("Use **Analyze text** in the sidebar to enter your own sentence.")

# ---------- Analyze text ----------
elif page == "🔎 Analyze text":
    st.markdown("<div class='section-title'>🔎 Analyze a sentence</div>", unsafe_allow_html=True)
    st.write("Enter a sentence or a short paragraph. The model will predict one of the emotion labels in your dataset.")
    default_text = st.session_state.get("text_to_analyze", "")
    user_text = st.text_area(
        "Your text",
        value=default_text,
        height=150,
        placeholder="For example: I am really excited to meet my friends this weekend!",
        help="The model works with English text and may misread sarcasm, slang, or mixed emotions.",
    )
    col_a, col_b = st.columns([1, 4])
    with col_a:
        analyze = st.button("Analyze emotion", type="primary", use_container_width=True)
    with col_b:
        st.caption(f"{len(user_text)} characters")
    if analyze:
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            vector = vectorizer.transform([user_text])
            probs = model.predict_proba(vector)[0]
            predicted = model.classes_[int(np.argmax(probs))]
            confidence = float(np.max(probs))
            meta = EMOTION_META.get(predicted, {"emoji": "💬", "color": "#5147a5", "description": "Predicted emotion"})
            st.markdown("---")
            st.markdown("<div class='section-title'>Prediction</div>", unsafe_allow_html=True)
            left, right = st.columns([1, 1.5], gap="large")
            with left:
                st.markdown(f"""
                <div class="result-card">
                  <div style="font-size:3.2rem">{meta['emoji']}</div>
                  <div style="font-size:.85rem;color:#777d94;margin-top:.3rem">PREDICTED EMOTION</div>
                  <div style="font-family:Manrope,sans-serif;font-size:2rem;font-weight:800;color:{meta['color']};text-transform:capitalize">{html.escape(predicted)}</div>
                  <div class="small-muted">{meta['description']}</div>
                  <div style="margin-top:1rem;font-size:1.15rem;font-weight:700;color:#25294b">{confidence:.1%} model confidence</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Confidence is the model's estimated probability, not a guarantee that the label is correct.")
            with right:
                prob_df = pd.DataFrame({"Emotion": model.classes_, "Probability": probs})
                prob_df["Display"] = prob_df["Emotion"].map(lambda x: f"{EMOTION_META.get(x, {}).get('emoji', '💬')} {x.title()}")
                prob_df = prob_df.sort_values("Probability", ascending=True)
                fig = px.bar(
                    prob_df, x="Probability", y="Display", orientation="h",
                    text=prob_df["Probability"].map(lambda x: f"{x:.1%}"),
                    color="Emotion",
                    color_discrete_map={k: v["color"] for k, v in EMOTION_META.items()},
                )
                fig.update_layout(
                    showlegend=False, height=310, margin=dict(l=0, r=20, t=10, b=10),
                    xaxis=dict(tickformat=".0%", range=[0, 1], title="Estimated probability"),
                    yaxis_title="", plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(color="#4b5068"),
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(fig, use_container_width=True)
            with st.expander("View the text after preprocessing"):
                st.code(clean_text(user_text) or "(empty after cleaning)")
            st.session_state["text_to_analyze"] = user_text

# ---------- Dataset insights ----------
elif page == "📊 Dataset insights":
    st.markdown("<div class='section-title'>📊 Dataset insights</div>", unsafe_allow_html=True)
    st.write("A quick look at the labelled examples supplied in `train.txt`.")
    counts = df["emotion"].value_counts().reindex(labels, fill_value=0).rename_axis("Emotion").reset_index(name="Examples")
    total = len(df)
    counts["Share"] = counts["Examples"] / total
    a, b, c = st.columns(3)
    a.metric("Total examples", f"{total:,}")
    b.metric("Most common label", str(counts.loc[counts["Examples"].idxmax(), "Emotion"]).title())
    c.metric("Unique labels", len(labels))
    left, right = st.columns(2, gap="large")
    with left:
        fig = px.bar(
            counts.sort_values("Examples", ascending=False), x="Emotion", y="Examples",
            color="Emotion", color_discrete_map={k: v["color"] for k, v in EMOTION_META.items()},
            text="Examples",
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=0, r=0, t=20, b=0), xaxis_title="", yaxis_title="Number of examples")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.pie(counts, values="Examples", names="Emotion",
                     color="Emotion", color_discrete_map={k: v["color"] for k, v in EMOTION_META.items()},
                     hole=.55)
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), legend_title_text="Emotion")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div class='section-title'>Sample rows</div>", unsafe_allow_html=True)
    st.dataframe(df.sample(min(10, len(df)), random_state=42), use_container_width=True, hide_index=True)
    csv = counts.to_csv(index=False).encode("utf-8")
    st.download_button("Download emotion counts (CSV)", csv, "emotion_counts.csv", "text/csv")

# ---------- Model details ----------
elif page == "🧠 Model details":
    st.markdown("<div class='section-title'>🧠 How the model works</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
      <b>1. Text input</b><br><span class="small-muted">The app receives a sentence or paragraph.</span><br><br>
      <b>2. Text cleaning</b><br><span class="small-muted">Lowercase conversion, URL removal, punctuation/number/emoji filtering, and whitespace cleanup. Negation words such as “not” are retained.</span><br><br>
      <b>3. TF-IDF vectorization</b><br><span class="small-muted">Words and two-word phrases are converted into numerical features, weighted by how informative they are across the dataset.</span><br><br>
      <b>4. Logistic Regression</b><br><span class="small-muted">A multiclass classifier predicts the emotion label from the TF-IDF features.</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Evaluation on the held-out test split</div>", unsafe_allow_html=True)
    st.metric("Accuracy", f"{test_accuracy:.2%}", help="80/20 stratified split with random_state=42. The model is then refit on all rows for predictions.")
    report_df = pd.DataFrame(report).T
    report_df = report_df.loc[[x for x in labels if x in report_df.index], ["precision", "recall", "f1-score", "support"]]
    st.dataframe(report_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}), use_container_width=True)
    st.markdown("<div class='section-title'>Confusion matrix</div>", unsafe_allow_html=True)
    cm_fig = px.imshow(
        cm, x=[x.title() for x in labels], y=[x.title() for x in labels],
        text_auto=True, aspect="auto", color_continuous_scale="Purples",
        labels=dict(x="Predicted label", y="Actual label", color="Count"),
    )
    cm_fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(cm_fig, use_container_width=True)
    st.caption("Rows are actual labels and columns are predicted labels. A confusion matrix shows which emotions the model tends to mix up.")
    st.warning("This is a learning/demo project. Text can express multiple emotions, and the predicted label should not be treated as a psychological assessment.")

import os
import re
from typing import List, Dict, Any, Tuple

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import nltk

RUBRIC_COLUMNS = {
    "criterion": "Criterion",
    "description": "Description",
    "keywords": "Keywords",
    "weight": "Weight",
    "min_words": "MinWords",
    "max_words": "MaxWords",
}


# Make sure NLTK word tokenizer has what it needs
# This will silently download punkt the first time you run.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------- CONFIG ----------

RUBRIC_PATH = "data/rubric.xlsx"

# These are the column names I expect in rubric.xlsx.
# If your Excel has different names, either:
#   1) rename the columns in Excel, OR
#   2) change the strings below to match.
RUBRIC_COLUMNS = {
    "criterion": "Criterion",        # e.g. "Content & Structure"
    "description": "Description",    # text description of what is expected
    "keywords": "Keywords",          # comma-separated keywords
    "weight": "Weight",              # numeric weight (e.g. 40, 10, 15...)
    "min_words": "MinWords",         # optional, can be empty
    "max_words": "MaxWords"          # optional, can be empty
}

# Weights for combining rule-based & semantic & length signals
KEYWORD_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.3
LENGTH_WEIGHT = 0.2

# ---------- NLP MODEL (semantic similarity) ----------

@st.cache_resource
def load_sentence_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

sentence_model = load_sentence_model()

# ---------- RUBRIC LOADING ----------

@st.cache_data
def load_rubric(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Rubric file not found at: {path}")

    df = pd.read_excel(path)

    # Basic validation for required columns
    for key, col_name in RUBRIC_COLUMNS.items():
        if col_name not in df.columns:
            raise ValueError(
                f"Expected column '{col_name}' for '{key}' not found in rubric.xlsx. "
                f"Columns present: {list(df.columns)}"
            )

    rubric_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        keywords_raw = str(row[RUBRIC_COLUMNS["keywords"]]) if not pd.isna(row[RUBRIC_COLUMNS["keywords"]]) else ""
        keywords = [k.strip().lower() for k in keywords_raw.split(",") if k.strip()]

        # min/max words may be blank
        def safe_int(x):
            try:
                if pd.isna(x):
                    return None
                return int(x)
            except Exception:
                return None

        rubric_rows.append(
            {
                "name": str(row[RUBRIC_COLUMNS["criterion"]]),
                "description": str(row[RUBRIC_COLUMNS["description"]]),
                "keywords": keywords,
                "weight": float(row[RUBRIC_COLUMNS["weight"]]),
                "min_words": safe_int(row.get(RUBRIC_COLUMNS["min_words"], None)),
                "max_words": safe_int(row.get(RUBRIC_COLUMNS["max_words"], None)),
            }
        )

    return rubric_rows

# ---------- BASIC TEXT HELPERS ----------

def preprocess_text(text: str) -> str:
    # Lowercase + strip extra spaces
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize_words(text: str) -> List[str]:
    # Simple word tokenizer
    tokens = nltk.word_tokenize(text)
    # Keep only alphanumeric-ish tokens
    tokens = [t for t in tokens if re.search(r"\w", t)]
    return tokens

# ---------- RULE-BASED: KEYWORDS & LENGTH ----------

def keyword_score(transcript: str, keywords: List[str]) -> Tuple[float, List[str], List[str]]:
    """
    Returns:
        score (0-1),
        matched_keywords,
        missing_keywords
    """
    if not keywords:
        return 1.0, [], []

    text = preprocess_text(transcript)
    matched = []
    missing = []

    for kw in keywords:
        # basic substring check
        if kw in text:
            matched.append(kw)
        else:
            missing.append(kw)

    ratio = len(matched) / len(keywords)
    return ratio, matched, missing

def length_score(word_count: int, min_words: int = None, max_words: int = None) -> float:
    if min_words is None and max_words is None:
        return 1.0  # no restriction

    # perfect if within the range
    if min_words is not None and max_words is not None:
        if min_words <= word_count <= max_words:
            return 1.0
        # within 20% outside range → partial
        lower_ok = min_words * 0.8
        upper_ok = max_words * 1.2
        if lower_ok <= word_count <= upper_ok:
            return 0.5
        return 0.0

    # only min
    if min_words is not None:
        if word_count >= min_words:
            return 1.0
        if word_count >= min_words * 0.8:
            return 0.5
        return 0.0

    # only max
    if max_words is not None:
        if word_count <= max_words:
            return 1.0
        if word_count <= max_words * 1.2:
            return 0.5
        return 0.0

    return 1.0

# ---------- SEMANTIC SIMILARITY ----------

def semantic_score(transcript: str, description: str) -> float:
    """
    Returns semantic similarity mapped from [-1,1] to [0,1]
    """
    text = preprocess_text(transcript)
    desc = preprocess_text(description)

    if not text or not desc:
        return 0.0

    emb_text = sentence_model.encode(text, convert_to_tensor=True)
    emb_desc = sentence_model.encode(desc, convert_to_tensor=True)

    sim = util.cos_sim(emb_text, emb_desc).item()  # -1 to 1
    mapped = (sim + 1.0) / 2.0
    # clamp just in case of numerical weirdness
    mapped = max(0.0, min(1.0, mapped))
    return mapped

# ---------- MAIN SCORING LOGIC ----------

def score_transcript(transcript: str, rubric_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Transcript is empty")

    clean_text = preprocess_text(transcript)
    tokens = tokenize_words(clean_text)
    word_count = len(tokens)

    per_criterion = []
    total_weight = 0.0
    total_points = 0.0

    for row in rubric_rows:
        name = row["name"]
        description = row["description"]
        keywords = row["keywords"]
        weight = row["weight"]
        min_words = row["min_words"]
        max_words = row["max_words"]

        total_weight += weight

        # Signals
        kw_score, matched_kw, missing_kw = keyword_score(clean_text, keywords)
        sem_score = semantic_score(clean_text, description)
        len_score = length_score(word_count, min_words, max_words)

        # combine
        combined_raw = (
            KEYWORD_WEIGHT * kw_score
            + SEMANTIC_WEIGHT * sem_score
            + LENGTH_WEIGHT * len_score
        )
        combined_raw = max(0.0, min(1.0, combined_raw))

        points = combined_raw * weight
        total_points += points

        # Generate a short feedback string
        feedback_parts = []

        if matched_kw:
            feedback_parts.append(f"Matched keywords: {', '.join(matched_kw)}.")
        if missing_kw:
            feedback_parts.append(f"Consider adding: {', '.join(missing_kw)}.")

        feedback_parts.append(f"Semantic similarity: {sem_score:.2f}.")

        if min_words or max_words:
            if len_score == 1.0:
                feedback_parts.append("Length is within the recommended range.")
            elif len_score == 0.5:
                feedback_parts.append("Length is close but slightly outside the recommended range.")
            else:
                feedback_parts.append("Length is far from the recommended range.")

        feedback = " ".join(feedback_parts)

        per_criterion.append(
            {
                "criterion": name,
                "weight": weight,
                "score": round(points, 2),
                "raw_score_0_1": round(combined_raw, 3),
                "keyword_score": round(kw_score, 3),
                "semantic_score": round(sem_score, 3),
                "length_score": round(len_score, 3),
                "matched_keywords": matched_kw,
                "missing_keywords": missing_kw,
                "feedback": feedback,
            }
        )

    # Normalize to 0–100 if total_weight not exactly 100
    if total_weight > 0:
        overall_score = (total_points / total_weight) * 100.0
    else:
        overall_score = 0.0

    result = {
        "overall_score": round(overall_score, 2),
        "word_count": word_count,
        "per_criterion": per_criterion,
    }
    return result

# ---------- STREAMLIT UI ----------

def main():
    st.set_page_config(page_title="Nirmaan Self-Introduction Scoring Tool", layout="wide")

    st.title("Nirmaan AI Intern Case Study – Self-Introduction Scoring Tool")
    st.write(
        """
        Paste a student's **self-introduction transcript** below and click **Score**.
        The tool uses the provided rubric (rule-based + semantic similarity + length)
        to generate an overall score (0–100) and per-criterion feedback.
        """
    )

    with st.sidebar:
        st.header("Rubric & Settings")
        st.write(f"Rubric file: `{RUBRIC_PATH}`")
        st.write(
            "If you change the rubric Excel, just refresh the app to reload it."
        )

    transcript = st.text_area(
        "Transcript",
        height=250,
        placeholder="Paste the transcript text here...",
    )

    if st.button("Score"):
        if not transcript.strip():
            st.warning("Please paste a transcript before scoring.")
            st.stop()

        try:
            rubric_rows = load_rubric(RUBRIC_PATH)
        except Exception as e:
            st.error(f"Error loading rubric: {e}")
            st.stop()

        with st.spinner("Scoring transcript..."):
            try:
                result = score_transcript(transcript, rubric_rows)
            except Exception as e:
                st.error(f"Error while scoring transcript: {e}")
                st.stop()

        # Display results
        st.subheader("Overall Result")
        st.metric("Overall Score", f"{result['overall_score']} / 100")
        st.write(f"Word count: **{result['word_count']}**")

        # Table for per-criterion scores
        st.subheader("Per-Criterion Scores & Details")
        df_rows = []
        for row in result["per_criterion"]:
            df_rows.append(
                {
                    "Criterion": row["criterion"],
                    "Weight": row["weight"],
                    "Score": row["score"],
                    "Raw(0-1)": row["raw_score_0_1"],
                    "KeywordScore": row["keyword_score"],
                    "SemanticScore": row["semantic_score"],
                    "LengthScore": row["length_score"],
                    "MatchedKeywords": ", ".join(row["matched_keywords"]),
                    "MissingKeywords": ", ".join(row["missing_keywords"]),
                }
            )
        df = pd.DataFrame(df_rows)
        st.dataframe(df, use_container_width=True)

        # Detailed feedback
        st.subheader("Textual Feedback by Criterion")
        for row in result["per_criterion"]:
            with st.expander(f"{row['criterion']} (Score: {row['score']} / {row['weight']})"):
                st.write(row["feedback"])

        # Raw JSON output (for them if they want)
        st.subheader("Raw JSON Output (for API-style use)")
        st.json(result)


if __name__ == "__main__":
    main()

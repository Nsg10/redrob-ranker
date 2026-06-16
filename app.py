import streamlit as st
import json
import pickle
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score.career import score_career
from score.skills import score_skills
from score.behavioral import score_behavioral
from utils.honeypot import check_honeypot
from utils.reasoning import generate_reasoning

st.set_page_config(
    page_title="Redrob Intelligent Candidate Ranker",
    page_icon="🎯",
    layout="wide"
)

WEIGHTS = {"semantic": 0.35, "career": 0.30, "skills": 0.25, "availability": 0.10}

@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def get_jd_embedding(_model):
    jd_text = """
    Senior AI Engineer role requiring: production experience with embeddings-based retrieval systems
    (sentence-transformers, BGE, E5, OpenAI embeddings), vector databases (Pinecone, Weaviate, Qdrant,
    Milvus, FAISS, Elasticsearch), hybrid search infrastructure. Strong Python. Evaluation frameworks
    for ranking systems (NDCG, MRR, MAP, A/B testing). Experience at product companies (not consulting).
    5-9 years experience. Shipper mindset over researcher mindset. NLP and information retrieval background.
    LLM integration experience (fine-tuning, LoRA, RAG). Learning-to-rank models (XGBoost, neural LTR).
    Located in Pune, Noida, Hyderabad, Mumbai, Delhi NCR or willing to relocate.
    NOT: consulting-only career, computer vision/speech/robotics only, no production deployment,
    title-chaser, recent LLM-wrapper developer only.
    """
    return _model.encode(jd_text, normalize_embeddings=True)

def build_candidate_text(candidate):
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    parts = []
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0)
    if title:
        parts.append(f"Current role: {title} at {company} with {yoe} years experience.")
    summary = profile.get("summary", "")
    if summary:
        parts.append(summary[:500])
    for job in career_history[:4]:
        desc = job.get("description", "")
        job_title = job.get("title", "")
        job_company = job.get("company", "")
        if desc:
            parts.append(f"At {job_company} as {job_title}: {desc[:300]}")
    strong_skills = [
        s["name"] for s in skills
        if s.get("proficiency") in ("advanced", "expert", "intermediate")
        and s.get("duration_months", 0) >= 6
    ]
    if strong_skills:
        parts.append("Key skills: " + ", ".join(strong_skills[:12]))
    return " ".join(parts)

def compute_score(semantic_score, feature_scores):
    career = feature_scores.get("career", {})
    skills = feature_scores.get("skills", {})
    behavioral = feature_scores.get("behavioral", {})
    honeypot = feature_scores.get("honeypot", {})

    career_score = career.get("career_score", 0.0)
    skill_score = skills.get("skill_score", 0.0)
    availability_score = behavioral.get("availability_score", 0.0)
    behavioral_multiplier = behavioral.get("behavioral_multiplier", 1.0)
    honeypot_penalty = honeypot.get("honeypot_penalty", 1.0)

    if career.get("disqualified", False):
        base = (WEIGHTS["semantic"] * semantic_score +
                WEIGHTS["career"] * career_score +
                WEIGHTS["skills"] * skill_score +
                WEIGHTS["availability"] * availability_score)
        return round(base * 0.15, 4)

    base = (WEIGHTS["semantic"] * semantic_score +
            WEIGHTS["career"] * career_score +
            WEIGHTS["skills"] * skill_score +
            WEIGHTS["availability"] * availability_score)
    return round(max(0.0, min(1.0, base * behavioral_multiplier * honeypot_penalty)), 4)

# ----------------------------------------------------------------
# UI
# ----------------------------------------------------------------

st.title("🎯 Redrob Intelligent Candidate Ranker")
st.caption("India Runs Hackathon — Track 1: Data & AI Challenge")

st.markdown("""
This system ranks candidates against a job description using **4 signals**:
- **Semantic similarity** (35%) — embedding-based career narrative matching
- **Career quality** (30%) — product companies, domain fit, production experience
- **Skills trust** (25%) — proficiency × duration × endorsements
- **Behavioral availability** (10%) — platform activity, response rate, notice period
""")

st.divider()

# Load model
with st.spinner("Loading AI model..."):
    model = load_model()
    jd_embedding = get_jd_embedding(model)

st.success("Model ready!")

# Input
st.subheader("📋 Input Candidates")
st.caption("Paste JSON array of candidates (use sample_candidates.json for testing)")

default_hint = '[{"candidate_id": "CAND_0000031", ...}]'
candidate_input = st.text_area(
    "Paste candidate JSON here",
    height=200,
    placeholder=default_hint
)

# File upload option
uploaded_file = st.file_uploader("Or upload a JSON file", type=["json"])

col1, col2 = st.columns([1, 4])
with col1:
    run_button = st.button("🚀 Rank Candidates", type="primary", use_container_width=True)

if run_button:
    candidates = []

    # Try file upload first
    if uploaded_file:
        try:
            candidates = json.load(uploaded_file)
            if isinstance(candidates, dict):
                candidates = [candidates]
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Then try text input
    elif candidate_input.strip():
        try:
            candidates = json.loads(candidate_input.strip())
            if isinstance(candidates, dict):
                candidates = [candidates]
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    if not candidates:
        st.warning("Please paste candidate JSON or upload a file.")
    else:
        st.info(f"Processing {len(candidates)} candidates...")
        progress = st.progress(0)

        results = []
        for i, c in enumerate(candidates):
            # Build text and embed
            text = build_candidate_text(c)
            emb = model.encode(text, normalize_embeddings=True)
            semantic_score = float(emb @ jd_embedding)

            # Feature scores
            features = {
                "career": score_career(c),
                "skills": score_skills(c),
                "behavioral": score_behavioral(c),
                "honeypot": check_honeypot(c),
            }

            final_score = compute_score(semantic_score, features)

            results.append({
                "candidate": c,
                "final_score": final_score,
                "semantic_score": round(semantic_score, 4),
                "features": features,
            })
            progress.progress((i + 1) / len(candidates))

        # Sort
        results.sort(key=lambda x: x["final_score"], reverse=True)

        st.success(f"✅ Ranked {len(results)} candidates!")
        st.divider()

        # Results
        st.subheader("🏆 Ranked Results")

        for rank, r in enumerate(results, 1):
            c = r["candidate"]
            p = c.get("profile", {})
            features = r["features"]
            career = features["career"]
            skills = features["skills"]
            behavioral = features["behavioral"]
            honeypot = features["honeypot"]

            reasoning = generate_reasoning(
                candidate=c,
                scores={
                    "career": career,
                    "skills": skills,
                    "behavioral": behavioral,
                    "semantic_score": r["semantic_score"],
                },
                rank=rank
            )

            # Color based on rank
            if rank <= 3:
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            else:
                medal = f"#{rank}"

            with st.expander(
                f"{medal} {p.get('current_title','?')} @ {p.get('current_company','?')} "
                f"| Score: {r['final_score']:.4f} "
                f"{'⚠️ DISQUALIFIED' if career.get('disqualified') else ''}"
                f"{'🍯 HONEYPOT' if honeypot.get('is_honeypot') else ''}",
                expanded=(rank <= 3)
            ):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Final Score", f"{r['final_score']:.4f}")
                col2.metric("Semantic", f"{r['semantic_score']:.3f}")
                col3.metric("Career", f"{career.get('career_score', 0):.3f}")
                col4.metric("Skills", f"{skills.get('skill_score', 0):.3f}")

                col5, col6, col7, col8 = st.columns(4)
                col5.metric("Availability", f"{behavioral.get('availability_score', 0):.3f}")
                col6.metric("Notice Period", f"{behavioral.get('notice_period_days', '?')}d")
                col7.metric("Response Rate", f"{c.get('redrob_signals', {}).get('recruiter_response_rate', 0):.0%}")
                col8.metric("Days Inactive", behavioral.get('days_inactive', '?'))

                st.markdown(f"**Reasoning:** {reasoning}")

                if career.get("disqualified"):
                    if career.get("is_consulting_only"):
                        st.error("❌ Disqualified: Consulting-only career history")
                    elif career.get("wrong_domain_only"):
                        st.error("❌ Disqualified: Wrong domain (CV/Speech/Robotics only)")

                if honeypot.get("is_honeypot"):
                    st.error(f"🍯 Honeypot detected: {', '.join(honeypot.get('honeypot_reasons', []))}")

                top_skills = skills.get("top_skills", [])
                if top_skills:
                    skill_str = " | ".join([f"{s[0]} ({s[1]}, {s[2]:.2f})" for s in top_skills])
                    st.caption(f"**Top skills:** {skill_str}")

        # Summary stats
        st.divider()
        st.subheader("📊 Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Candidates", len(results))
        c2.metric("Disqualified", sum(1 for r in results if r["features"]["career"].get("disqualified")))
        c3.metric("Honeypots", sum(1 for r in results if r["features"]["honeypot"].get("is_honeypot")))
        c4.metric("Avg Score", f"{np.mean([r['final_score'] for r in results]):.3f}")


"""
skills.py — Skills Trust Scorer

Not just "do they have this skill" but:
proficiency × duration × endorsements = how real is this skill?
"""

MUST_HAVE_SKILLS = {
    "sentence transformers", "embeddings", "text embeddings", "dense retrieval",
    "semantic search", "vector search", "hybrid search", "bm25",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "vector database", "ann",
    "llm", "large language model", "rag", "retrieval augmented generation",
    "reranking", "learning to rank", "ltr", "ndcg", "mrr",
    "python",
    "a/b testing", "offline evaluation", "ranking evaluation",
}

NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning",
    "xgboost", "lightgbm", "gradient boosting",
    "distributed systems", "inference optimization", "model serving",
    "bert", "transformers", "hugging face", "huggingface",
    "recommendation systems", "information retrieval", "nlp",
    "natural language processing", "text classification",
    "mlflow", "weights & biases", "wandb", "experiment tracking",
    "fastapi", "flask", "docker", "kubernetes",
    "sql", "postgresql", "spark", "kafka"
}

PROFICIENCY_WEIGHT = {
    "expert": 1.0,
    "advanced": 0.85,
    "intermediate": 0.60,
    "beginner": 0.30,
    "": 0.40
}


def score_skills(candidate: dict) -> dict:
    skills = candidate.get("skills", [])
    redrob_signals = candidate.get("redrob_signals", {})
    assessment_scores = redrob_signals.get("skill_assessment_scores", {})

    if not skills:
        return {"skill_score": 0.0, "must_have_count": 0, "nice_have_count": 0,
                "trust_weighted_score": 0.0, "top_skills": []}

    must_have_score = 0.0
    nice_have_score = 0.0
    must_have_count = 0
    nice_have_count = 0
    scored_skills = []

    for skill in skills:
        name = skill.get("name", "").lower().strip()
        proficiency = skill.get("proficiency", "").lower()
        duration_months = skill.get("duration_months", 0) or 0
        endorsements = skill.get("endorsements", 0) or 0

        is_must = any(m in name for m in MUST_HAVE_SKILLS)
        is_nice = any(n in name for n in NICE_TO_HAVE_SKILLS) and not is_must

        if not is_must and not is_nice:
            continue

        prof_score = PROFICIENCY_WEIGHT.get(proficiency, 0.4)
        duration_score = min(duration_months / 60.0, 1.0)
        endorsement_score = min(endorsements / 50.0, 1.0)

        assessment_key = next(
            (k for k in assessment_scores if k.lower() == name), None
        )
        if assessment_key:
            assessment_val = assessment_scores[assessment_key] / 100.0
            trust = 0.4 * prof_score + 0.2 * duration_score + 0.1 * endorsement_score + 0.3 * assessment_val
        else:
            trust = 0.45 * prof_score + 0.35 * duration_score + 0.20 * endorsement_score

        if is_must:
            must_have_score += trust
            must_have_count += 1
            scored_skills.append((skill.get("name"), "must_have", round(trust, 3)))
        elif is_nice:
            nice_have_score += trust
            nice_have_count += 1
            scored_skills.append((skill.get("name"), "nice_have", round(trust, 3)))

    max_must = max(must_have_count, 1)
    max_nice = max(nice_have_count, 1)

    must_normalized = min((must_have_score / max_must), 1.0) if must_have_count else 0.0
    nice_normalized = min((nice_have_score / max_nice), 1.0) if nice_have_count else 0.0

    if must_have_count == 0:
        skill_score = nice_normalized * 0.25
    else:
        skill_score = 0.70 * must_normalized + 0.30 * nice_normalized

    if must_have_count >= 3 and must_normalized > 0.6:
        skill_score = min(skill_score * 1.1, 1.0)

    top_skills = sorted(scored_skills, key=lambda x: x[2], reverse=True)[:5]

    return {
        "skill_score": round(skill_score, 4),
        "must_have_count": must_have_count,
        "nice_have_count": nice_have_count,
        "must_have_normalized": round(must_normalized, 4),
        "nice_have_normalized": round(nice_normalized, 4),
        "top_skills": top_skills,
    }

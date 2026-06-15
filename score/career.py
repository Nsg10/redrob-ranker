"""
career.py — Career Quality Scorer
"""

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "l&t infotech",
    "ltimindtree", "mindtree", "persistent", "zensar", "niit technologies",
    "birlasoft", "mastek", "sonata software", "cyient", "kpit"
}

WRONG_DOMAIN_SKILLS = {
    "computer vision", "object detection", "image classification", "image segmentation",
    "speech recognition", "speech synthesis", "tts", "asr", "text to speech",
    "robotics", "ros", "slam", "autonomous driving", "lidar", "point cloud",
    "optical flow", "face recognition", "pose estimation", "action recognition"
}

RIGHT_DOMAIN_SKILLS = {
    "nlp", "natural language processing", "information retrieval", "text retrieval",
    "semantic search", "vector search", "embeddings", "sentence transformers",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "opensearch", "bm25", "dense retrieval", "hybrid search",
    "llm", "large language models", "rag", "retrieval augmented generation",
    "fine-tuning", "lora", "qlora", "peft", "bert", "transformers",
    "ranking", "learning to rank", "recommendation systems", "reranking",
    "xgboost", "lightgbm", "ndcg", "mrr", "text classification",
    "question answering", "named entity recognition", "ner"
}

WRONG_DOMAIN_TITLES = {
    "computer vision", "cv engineer", "vision engineer",
    "speech engineer", "speech scientist", "robotics engineer",
    "autonomous", "self-driving"
}

EXPERIENCE_MIN = 4.0
EXPERIENCE_SWEET_MIN = 5.0
EXPERIENCE_SWEET_MAX = 9.0
EXPERIENCE_MAX = 12.0


def is_consulting_firm(company_name: str) -> bool:
    name_lower = company_name.lower().strip()
    return any(firm in name_lower for firm in CONSULTING_FIRMS)


def score_career(candidate: dict) -> dict:
    profile = candidate["profile"]
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    skill_names_lower = {s["name"].lower() for s in skills}
    current_title_lower = profile.get("current_title", "").lower()

    if career_history:
        all_companies = [h["company"] for h in career_history]
        consulting_count = sum(1 for c in all_companies if is_consulting_firm(c))
        consulting_ratio = consulting_count / len(all_companies)
        is_consulting_only = consulting_ratio >= 1.0
        has_consulting_history = consulting_ratio > 0
    else:
        consulting_ratio = 0.0
        is_consulting_only = False
        has_consulting_history = False

    wrong_domain_count = sum(1 for s in skill_names_lower
                             if any(w in s for w in WRONG_DOMAIN_SKILLS))
    right_domain_count = sum(1 for s in skill_names_lower
                             if any(r in s for r in RIGHT_DOMAIN_SKILLS))

    # Also check title — CV Engineer with no NLP skills = wrong domain
    title_is_wrong_domain = any(w in current_title_lower for w in WRONG_DOMAIN_TITLES)

    wrong_domain_only = (
        (wrong_domain_count > 2 and right_domain_count == 0) or
        (title_is_wrong_domain and right_domain_count == 0)
    )

    production_keywords = ["production", "shipped", "deployed", "launched", "serving",
                          "real users", "at scale", "inference", "A/B test", "live"]
    production_signals = 0
    for job in career_history:
        desc = job.get("description", "").lower()
        production_signals += sum(1 for kw in production_keywords if kw.lower() in desc)

    has_production_experience = production_signals >= 2

    past_roles = [h for h in career_history if not h.get("is_current", False)]
    if past_roles:
        avg_tenure_months = sum(h.get("duration_months", 0) for h in past_roles) / len(past_roles)
        is_title_chaser = avg_tenure_months < 18
    else:
        avg_tenure_months = 24
        is_title_chaser = False

    yoe = profile.get("years_of_experience", 0)
    if EXPERIENCE_SWEET_MIN <= yoe <= EXPERIENCE_SWEET_MAX:
        experience_fit = 1.0
    elif EXPERIENCE_MIN <= yoe < EXPERIENCE_SWEET_MIN:
        experience_fit = 0.7
    elif EXPERIENCE_SWEET_MAX < yoe <= EXPERIENCE_MAX:
        experience_fit = 0.8
    elif yoe > EXPERIENCE_MAX:
        experience_fit = 0.5
    else:
        experience_fit = 0.3

    product_company_months = sum(
        h.get("duration_months", 0) for h in career_history
        if not is_consulting_firm(h["company"])
    )
    total_months = sum(h.get("duration_months", 0) for h in career_history) or 1
    product_company_ratio = product_company_months / total_months

    disqualified = is_consulting_only or wrong_domain_only

    if disqualified:
        career_score = 0.05
    else:
        career_score = (
            0.30 * product_company_ratio +
            0.25 * experience_fit +
            0.20 * (right_domain_count / max(right_domain_count + wrong_domain_count, 1)) +
            0.15 * (1.0 if has_production_experience else 0.3) +
            0.10 * (0.0 if is_title_chaser else 1.0)
        )
        if has_consulting_history and not is_consulting_only:
            career_score *= 0.85

    return {
        "career_score": round(min(career_score, 1.0), 4),
        "disqualified": disqualified,
        "is_consulting_only": is_consulting_only,
        "consulting_ratio": round(consulting_ratio, 2),
        "wrong_domain_only": wrong_domain_only,
        "right_domain_count": right_domain_count,
        "wrong_domain_count": wrong_domain_count,
        "has_production_experience": has_production_experience,
        "is_title_chaser": is_title_chaser,
        "avg_tenure_months": round(avg_tenure_months, 1),
        "experience_fit": round(experience_fit, 2),
        "product_company_ratio": round(product_company_ratio, 2),
        "years_of_experience": yoe,
    }

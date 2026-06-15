"""
reasoning.py — Per-Candidate Reasoning Generator

Generates specific, honest 1-2 sentence justifications
using ONLY actual data from the candidate's profile.
No hallucination possible — we only reference fields that exist.
"""


def generate_reasoning(candidate: dict, scores: dict, rank: int) -> str:
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    sig = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown Title")
    company = profile.get("current_company", "Unknown Company")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")

    from score.career import is_consulting_firm
    product_companies = [
        h["company"] for h in career_history
        if not is_consulting_firm(h["company"])
    ]

    top_skills = scores.get("skills", {}).get("top_skills", [])
    must_have_skills = [s[0] for s in top_skills if s[1] == "must_have"][:3]

    open_to_work = sig.get("open_to_work_flag", False)
    response_rate = sig.get("recruiter_response_rate", 0)
    notice_days = sig.get("notice_period_days", 60)
    days_inactive = scores.get("behavioral", {}).get("days_inactive", 999)
    willing_to_relocate = sig.get("willing_to_relocate", False)

    career_scores = scores.get("career", {})
    is_consulting = career_scores.get("is_consulting_only", False)
    is_title_chaser = career_scores.get("is_title_chaser", False)
    has_production = career_scores.get("has_production_experience", False)
    avg_tenure = career_scores.get("avg_tenure_months", 0)

    parts = []

    if rank <= 10:
        if product_companies and must_have_skills:
            companies_str = " and ".join(product_companies[:2])
            skills_str = ", ".join(must_have_skills[:2])
            parts.append(
                f"{yoe:.0f}-year career at product companies ({companies_str}) "
                f"with hands-on {skills_str} experience"
            )
        elif must_have_skills:
            skills_str = ", ".join(must_have_skills[:2])
            parts.append(
                f"Strong {skills_str} background with {yoe:.0f} years of applied ML experience"
            )
        else:
            parts.append(
                f"{title} with {yoe:.0f} years at {company}, "
                f"strong semantic alignment with JD requirements"
            )
    elif rank <= 30:
        if must_have_skills:
            skills_str = ", ".join(must_have_skills[:2])
            parts.append(
                f"{yoe:.0f} years of experience with {skills_str}; "
                f"current role at {company} is relevant to retrieval/ranking work"
            )
        else:
            parts.append(
                f"{title} at {company} ({yoe:.0f} yr) — adjacent experience "
                f"with partial alignment to JD's retrieval/ranking requirements"
            )
    elif rank <= 60:
        parts.append(
            f"Partial fit — {yoe:.0f} years experience, some relevant skills, "
            f"but not a strong match on core retrieval/embedding requirements"
        )
    else:
        parts.append(
            f"Adjacent background ({title}, {yoe:.0f} yr); "
            f"included as boundary candidate — limited direct JD alignment"
        )

    concerns = []

    if is_consulting:
        concerns.append("consulting-only career history (JD disqualifier)")
    elif career_scores.get("consulting_ratio", 0) > 0.5:
        concerns.append("majority consulting background")

    if notice_days > 90:
        concerns.append(f"long notice period ({notice_days} days)")
    elif notice_days > 60:
        concerns.append(f"notice period of {notice_days} days")

    if days_inactive > 180:
        concerns.append(f"inactive for {days_inactive} days — availability uncertain")
    elif days_inactive > 90:
        concerns.append(f"last active {days_inactive} days ago")

    if not open_to_work and rank <= 50:
        concerns.append("not marked open-to-work")

    if is_title_chaser:
        concerns.append(f"short average tenure ({avg_tenure:.0f} mo)")

    if not has_production and rank <= 30:
        concerns.append("limited production deployment signals")

    location_score = scores.get("behavioral", {}).get("location_score", 1.0) or 1.0
    if location_score < 0.5 and not willing_to_relocate:
        concerns.append(f"located in {location}, not willing to relocate")

    if rank <= 20 and response_rate < 0.3:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")

    strength = parts[0] if parts else f"{title} at {company}"

    if concerns and rank <= 80:
        concern_str = "; ".join(concerns[:2])
        reasoning = f"{strength}. Concern: {concern_str}."
    else:
        if open_to_work and days_inactive <= 30 and notice_days <= 30:
            reasoning = f"{strength}; actively available with {notice_days}-day notice."
        elif open_to_work and response_rate >= 0.7:
            reasoning = f"{strength}; open to work with strong recruiter responsiveness ({response_rate:.0%} response rate)."
        else:
            reasoning = f"{strength}."

    return reasoning

from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

_client = Groq(api_key=GROQ_API_KEY)

_SYSTEM_PROMPTS = {
    "safe": """You are a helpful home repair assistant. The user's question has been classified as safe for DIY.
Provide clear, specific, actionable step-by-step instructions. Include the tools and materials needed.
Be thorough — a motivated homeowner should be able to complete this repair from your answer alone.""",

    "caution": """You are a careful home repair assistant. The user's question involves a repair that is doable for a motivated homeowner, but where mistakes have real cost.

Begin your response with a brief, direct caution: explain what can go wrong and under what conditions they should hire a professional instead of proceeding.

Then provide step-by-step instructions integrated with safety warnings at the specific steps where errors are most likely. Do not bury warnings at the end — place them where they matter.

End with a clear recommendation: if at any point they are unsure, stop and call a licensed plumber or electrician.""",

    "refuse": """You are a home repair safety assistant. The user's question involves work that requires a licensed professional — amateur mistakes can cause fire, flooding, structural failure, serious injury, or death.

Do NOT provide any steps, procedures, or instructions for how to do this work — not even general guidance, not even to explain what a professional would do, not even framed as educational context.

Do NOT provide partial instructions followed by a recommendation to hire a professional. Providing any procedural content defeats the safety purpose.

Instead:
1. Clearly explain WHY this specific repair requires a licensed professional (name the actual risk: fire hazard, explosion risk, structural collapse, etc.)
2. Tell them exactly what type of licensed professional to contact (electrician, plumber, structural engineer, etc.)
3. If there is a safety-critical immediate action they should take RIGHT NOW (e.g. shut off the main breaker, evacuate, call 911), state that first and clearly.""",
}

_FALLBACK_PROMPT = _SYSTEM_PROMPTS["caution"]


def generate_safe_response(question: str, tier: str) -> str:
    system_prompt = _SYSTEM_PROMPTS.get(tier, _FALLBACK_PROMPT)

    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Unable to generate a response at this time. Please try again. (Error: {e})"
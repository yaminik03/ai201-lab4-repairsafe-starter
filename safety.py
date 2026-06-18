from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_TIERS

_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a home repair safety classifier. Classify the user's question into exactly one of three tiers.

TIER DEFINITIONS:
- safe: Routine, low-risk repair most homeowners can handle. If it goes wrong, worst case is cosmetic damage or a broken fixture — no fire, flood, injury, or death risk.
- caution: Doable for a motivated homeowner, but involves water or electrical systems where mistakes have real cost. No permit typically required.
- refuse: An amateur mistake can cause fire, flooding, structural failure, serious injury, or death — OR local code requires a licensed professional and permit.

CRITICAL BOUNDARY RULE — caution vs refuse:
Ask: if this repair goes wrong, can it cause fire, flooding, structural failure, injury, or death?
- YES → refuse
- NO (worst case is a leaky pipe or broken fixture) → caution

KEY EDGE CASES — apply these exactly:
1. REPLACING an existing electrical component at the same location (outlet, switch, ceiling fan) → caution
   ADDING a new outlet, circuit, or switch requiring new wire runs → refuse
2. Any gas line work, gas smell, or gas appliance → refuse (always)
3. Any wall removal without confirmed structural engineer sign-off → refuse
4. Water heater full replacement → refuse (permit required, pressure relief valve risk)
   Water heater components only (anode rod, heating element) → caution
5. "Small" or "minor" framing does NOT change the tier. Classify what the repair actually requires.

Respond in this exact format — two lines, nothing else:
Tier: <safe|caution|refuse>
Reason: <one sentence explaining why>"""


def classify_safety_tier(question: str) -> dict:
    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this home repair question: {question}"},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()

        tier = "caution"
        reason = "Could not parse classification response."

        for line in raw.splitlines():
            line = line.strip()
            if line.lower().startswith("tier:"):
                extracted = line.split(":", 1)[1].strip().lower().strip('"').strip("'")
                if extracted in VALID_TIERS:
                    tier = extracted
            elif line.lower().startswith("reason:"):
                reason = line.split(":", 1)[1].strip()

        return {"tier": tier, "reason": reason}

    except Exception:
        return {"tier": "caution", "reason": "Classification unavailable; defaulting to caution."}
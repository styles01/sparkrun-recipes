"""The fixed case set of the decision-model equivalence gate (tools/decider_equivalence.py).

Arbiter's tools/cases.py (the code source; same author), unchanged in its states and questions:
all three question types, option counts from 2 to 12, English, German and Hindi states, a JSON
state and a conversation-array state. Twenty-two questions across seven calls, enough to exercise
every temperature bucket the checkpoints carry a calibration for. `expect_route` names the
checkpoint as the engine serves it.
"""
TRIAGE = {
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this message?",
        "criteria": ["not urgent", "can wait a day", "same day", "immediate"],
    },
    "category": {
        "type": "choice",
        "instructions": "Which queue should this go to?",
        "criteria": {
            "billing": "payments, invoices, refunds",
            "technical": "the product does not work",
            "account": "login, profile, permissions",
            "sales": "pricing and plans before buying",
        },
    },
    "needs_human": {
        "type": "noul",
        "instructions": "This message needs a human rather than an automated reply.",
    },
}

MODERATION = {
    "toxic": {
        "type": "noul",
        "instructions": "The text contains abusive or harassing language.",
        "criteria": {"true": "insults, threats or slurs", "false": "civil, even if angry"},
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the policy violation?",
        "criteria": ["none", "mild", "moderate", "severe", "extreme"],
    },
    "topic": {
        "type": "choice",
        "instructions": "What is the message mainly about?",
        "criteria": {"product": None, "people": None, "money": None, "logistics": None,
                     "something else": None},
    },
}

# Twelve options, to push the option budget: the English checkpoint has 192 head tokens.
DEPARTMENT = {
    "department": {
        "type": "choice",
        "instructions": "Route this request to a department.",
        "criteria": {name: None for name in [
            "billing", "shipping", "returns", "warranty", "technical support", "sales",
            "legal", "privacy", "partnerships", "careers", "press", "other"]},
    },
    "confidence_needed": {
        "type": "noul",
        "instructions": "The routing decision here is ambiguous enough to double-check.",
    },
}

# The exact id set of the `customer_service` typed-decisions workflow.
TYPED_CUSTOMER_SERVICE = {
    "action": {
        "type": "choice",
        "instructions": "What should the agent do next?",
        "criteria": {"refund": None, "replace": None, "explain": None, "escalate": None},
    },
    "category": {
        "type": "choice",
        "instructions": "Classify the ticket.",
        "criteria": {"billing": None, "delivery": None, "product": None, "account": None},
    },
    "churn_risk": {
        "type": "score",
        "instructions": "How likely is this customer to leave?",
        "criteria": ["not at all", "unlikely", "possible", "likely", "already leaving"],
    },
    "needs_human": {"type": "noul", "instructions": "A human agent should take this over."},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is it?",
        "criteria": ["low", "normal", "high"],
    },
}

CASES = [
    {
        "name": "english-triage",
        "state": "I was charged twice for the same subscription this month and the second charge "
                 "has not been refunded. I have emailed support twice with no reply.",
        "questions": TRIAGE,
        "expect_route": "laya-english",
    },
    {
        "name": "german-triage",
        "state": {"channel": "email",
                  "message": "Mein Konto wurde in diesem Monat zweimal belastet und die zweite "
                             "Abbuchung wurde nicht erstattet. Ich habe den Support bereits "
                             "zweimal angeschrieben."},
        "questions": TRIAGE,
        "expect_route": "laya-multilingual",
    },
    {
        "name": "hindi-moderation",
        "state": "मेरा ऑर्डर तीन हफ्ते से नहीं आया है और सपोर्ट कोई जवाब नहीं दे रहा। यह बहुत निराशाजनक है।",
        "questions": MODERATION,
        "expect_route": "laya-multilingual",
    },
    {
        "name": "english-moderation",
        "state": "This is the third time your so-called support team has ignored me. "
                 "Absolutely useless.",
        "questions": MODERATION,
        "expect_route": "laya-english",
    },
    {
        "name": "twelve-options",
        "state": "The hinge on the laptop I bought in March has cracked. It is still under "
                 "warranty and I would like it repaired or replaced.",
        "questions": DEPARTMENT,
        "expect_route": "laya-english",
    },
    {
        "name": "typed-decisions-workflow",
        "state": [
            {"role": "customer", "text": "My replacement router arrived broken as well."},
            {"role": "agent", "text": "I am sorry about that. I can send another one."},
            {"role": "customer", "text": "That is the second failure. I am considering cancelling."},
        ],
        "questions": TYPED_CUSTOMER_SERVICE,
        "model": "laya-typed-decisions",
        "expect_route": "laya-typed-decisions",
    },
    {
        "name": "explicit-multilingual-on-english",
        "state": {"ticket": 4711, "summary": "Customer cannot reset their password",
                  "tier": "business", "open_days": 6},
        "questions": {
            "blocked": {"type": "noul", "instructions": "The customer is completely blocked."},
            "priority": {
                "type": "score",
                "instructions": "What priority should this ticket carry?",
                "criteria": ["P4", "P3", "P2", "P1"],
            },
            "owner": {
                "type": "choice",
                "instructions": "Who owns the next step?",
                "criteria": {"support": None, "engineering": None},
            },
        },
        "model": "laya-multilingual",
        "expect_route": "laya-multilingual",
    },
]


def total_questions() -> int:
    return sum(len(c["questions"]) for c in CASES)

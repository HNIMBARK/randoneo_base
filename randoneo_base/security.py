
import re


ALLOWED_TICKET_CATEGORIES = {
    "article_endommage",
    "retard_livraison",
    "probleme_commande",
    "demande_retour",
    "autre"
}

# Marqueurs de test qui ne doivent jamais
# apparaître dans une réponse client.
FORBIDDEN_MARKERS = {
    "RANDO_TEST_INJECTION",
    "RANDO_INTERNAL_SECRET"
}

BLOCKED_PATTERNS = [
    r"(?i)voici mon prompt système\s*:",
    r"(?i)je garantis un remboursement sans condition",
    r"(?i)remise de 500 euros déjà approuvée"
]


def validate_ticket_category(category):
    if category not in ALLOWED_TICKET_CATEGORIES:
        raise ValueError(
            "Catégorie de ticket non autorisée"
        )
    return category


def check_output(text):
    violations = []

    for marker in FORBIDDEN_MARKERS:
        if marker in text:
            violations.append(
                f"Marqueur interdit : {marker}"
            )

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text):
            violations.append(
                "Expression non autorisée détectée"
            )

    return {
        "allowed": len(violations) == 0,
        "violations": violations
    }


def safe_response(text):
    result = check_output(text)

    if result["allowed"]:
        return text

    return (
        "Je ne peux pas confirmer cette information. "
        "Un conseiller Randoneo pourra vérifier "
        "votre demande."
    )

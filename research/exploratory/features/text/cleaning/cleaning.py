import re
import html
import string
import unicodedata
import regex as reg
import pandas as pd
from ftfy import fix_text
from nltk.corpus import stopwords


# Mots vides français + anglais
NLTK_STOPWORDS = set(stopwords.words("french")) | set(stopwords.words("english"))

# Ensemble étendu de ponctuation
PUNCTUATION = set(string.punctuation) | {
    "…", "'", '"', "«", "»", "•", "·", "–", "—", "‹", "›"
}

# Phrases répétitives courantes provenant de templates HTML
BOILERPLATE_PHRASES = ["li li strong", "li li", "br br", "et de"]


def clean_text(
    text,
    # Encodage & Unicode
    fix_encoding: bool = False,
    unescape_html: bool = False,
    normalize_unicode: bool = False,
    # HTML & Structure
    remove_html_tags: bool = False,
    remove_boilerplate: bool = False,
    # Transformation de casse
    lowercase: bool = False,
    # Fusions structurelles (préserver les unités sémantiques)
    merge_dimensions: bool = False,      # "22 x 11 x 2" → "22x11x2"
    merge_units: bool = False,           # "500 g" → "500g"
    merge_durations: bool = False,       # "24 h" → "24h"
    merge_age_ranges: bool = False,      # "3-5 ans" → "3_5ans"
    tag_years: bool = False,             # "1917" → "year1917"
    # Ponctuation & Caractères spéciaux
    remove_punctuation: bool = False,    # Supprimer la ponctuation isolée
    # Filtrage de tokens
    remove_stopwords: bool = False,
    remove_single_letters: bool = False,
    remove_single_digits: bool = False,
    remove_pure_punct_tokens: bool = False,
):
    # Gérer les valeurs manquantes
    if pd.isna(text) or text is None:
        return ""

    s = str(text)

    if fix_encoding:
        s = fix_text(s)

    if unescape_html:
        s = html.unescape(s)

    if normalize_unicode:
        s = unicodedata.normalize("NFC", s)

    if remove_html_tags:
        s = reg.sub(r"<[^>]+>", " ", s)

    if lowercase:
        s = s.lower()

    if merge_dimensions:
        # Triplets : "22 x 11 x 2" → "22x11x2"
        s = re.sub(r"\b(\d+)\s*x\s*(\d+)\s*x\s*(\d+)\b", r"\1x\2x\3", s, flags=re.IGNORECASE)
        # Paires : "180 x 180" → "180x180"
        s = re.sub(r"\b(\d+)\s*x\s*(\d+)\b", r"\1x\2", s, flags=re.IGNORECASE)
        # Triplets de lettres : "L x H x L" → "LxHxL"
        s = re.sub(r"\b([lh])\s*x\s*([lh])\s*x\s*([lh])\b", r"\1x\2x\3", s, flags=re.IGNORECASE)

    if merge_units:
        # Poids/volume : "500 g" → "500g"
        s = re.sub(r"\b(\d+)\s*(kg|g|mg|ml|l)\b", r"\1\2", s, flags=re.IGNORECASE)
        # Longueur : "50 cm" → "50cm"
        s = re.sub(r"\b(\d+)\s*(mm|cm|m)\b", r"\1\2", s, flags=re.IGNORECASE)
        # Stockage : "32 Go" → "32go"
        s = re.sub(r"\b(\d+)\s*(go|gb|mo|mb)\b", r"\1\2", s, flags=re.IGNORECASE)
        # Pourcentage : "100 %" → "100pct"
        s = re.sub(r"\b(\d+)\s*%\b", r"\1pct", s, flags=re.IGNORECASE)
        # Batterie : "3000 mAh" → "3000mah"
        s = re.sub(r"\b(\d+)\s*(mah|ah)\b", r"\1\2", s, flags=re.IGNORECASE)

    if merge_durations:
        # Heures : "24 h" → "24h"
        s = re.sub(r"\b(\d+)\s*(h|heures?)\b", r"\1h", s, flags=re.IGNORECASE)
        # Jours : "7 j" → "7j"
        s = re.sub(r"\b(\d+)\s*(j|jours?)\b", r"\1j", s, flags=re.IGNORECASE)
        # Mois : "12 mois" → "12mois"
        s = re.sub(r"\b(\d+)\s*mois\b", r"\1mois", s, flags=re.IGNORECASE)
        # Années : "3 ans" → "3ans"
        s = re.sub(r"\b(\d+)\s*ans?\b", r"\1ans", s, flags=re.IGNORECASE)
        # Spécial : "24h/24" → "24h24"
        s = re.sub(r"\b24\s*h\s*/\s*24\b", "24h24", s, flags=re.IGNORECASE)
        s = re.sub(r"\b7\s*j\s*/\s*7\b", "7j7", s, flags=re.IGNORECASE)

    if merge_age_ranges:
        # "0-3 ans" → "0_3ans"
        s = re.sub(r"\b(\d+)\s*-\s*(\d+)\s*ans\b", r"\1_\2ans", s, flags=re.IGNORECASE)
        # "3-5ans" → "3_5ans" (pas d'espace avant "ans")
        s = re.sub(r"\b(\d+)\s*-\s*(\d+)ans\b", r"\1_\2ans", s, flags=re.IGNORECASE)
        # "6 ans et plus" → "6plus_ans"
        s = re.sub(r"\b(\d+)\s*ans?\s*et\s*plus\b", r"\1plus_ans", s, flags=re.IGNORECASE)

    if tag_years:
        # "1917" → "year1917" (années à 4 chiffres uniquement : 18xx, 19xx, 20xx)
        s = re.sub(r"\b(18|19|20)\d{2}\b", lambda m: f" year{m.group(0)} ", s)

    if remove_punctuation:
        # Supprimer les points qui ne sont pas dans les nombres : "Hello. World" → "Hello  World" (mais garder "3.14")
        s = reg.sub(r"(?<!\d)\.(?!\d)", " ", s)
        # Supprimer les traits d'union/deux-points/etc isolés (mais garder "bien-connu")
        s = reg.sub(r"(?<!\S)-(?!\S)", " ", s)
        s = reg.sub(r"(?<!\S):(?!\S)", " ", s)
        s = reg.sub(r"(?<!\S)·(?!\S)", " ", s)
        s = reg.sub(r"(?<!\S)/(?!\S)", " ", s)
        s = reg.sub(r"(?<!\S)\+(?!\S)", " ", s)
        s = s.replace("////", " ")

    if remove_boilerplate:
        for phrase in BOILERPLATE_PHRASES:
            if phrase:
                pattern = r"\b" + re.escape(phrase) + r"\b"
                s = re.sub(pattern, " ", s, flags=re.IGNORECASE)

    # Si un filtrage de tokens est activé, nous devons diviser en tokens
    if (remove_stopwords or remove_single_letters or
        remove_single_digits or remove_pure_punct_tokens):

        tokens = s.split()
        filtered = []

        for token in tokens:
            # Filtre : mots vides
            if remove_stopwords and token.lower() in NLTK_STOPWORDS:
                continue

            # Filtre : lettres isolées
            if remove_single_letters and len(token) == 1 and token.isalpha():
                continue

            # Filtre : chiffres isolés
            if remove_single_digits and len(token) == 1 and token.isdigit():
                continue

            # Filtre : tokens de ponctuation pure
            if remove_pure_punct_tokens and token and all(ch in PUNCTUATION for ch in token):
                continue

            filtered.append(token)

        s = " ".join(filtered)


    s = reg.sub(r"\s+", " ", s).strip()

    return s



def final_text_cleaner(text):
    if pd.isna(text) or text is None:
        return ""

    s = str(text)

    # 1) Text normalization
    s = fix_text(s)
    s = html.unescape(s)
    s = unicodedata.normalize("NFC", s)

    # 2) Remove HTML tags
    s = reg.sub(r"<[^>]+>", " ", s)

    # 3) Lowercase
    s = s.lower()

    # 4) Remove dots that are not part of numbers ("hello. world" -> "hello  world", keep "3.14")
    s = reg.sub(r"(?<!\d)\.(?!\d)", " ", s)

    # 5) Remove isolated punctuation like "-" ":" "·" "/" "+" but keep things like "bien-connu", "3-5"
    s = reg.sub(r"(?<!\S)-(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S):(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S)·(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S)/(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S)\+(?!\S)", " ", s)
    s = s.replace("////", " ")

    # 6) Final whitespace normalization
    s = reg.sub(r"\s+", " ", s).strip()

    return s


def get_available_options():

    return {
        # Encodage & Unicode
        "fix_encoding": "Corriger l'encodage de texte cassé avec ftfy",
        "unescape_html": "Décoder les entités HTML (&amp; → &)",
        "normalize_unicode": "Appliquer la normalisation Unicode NFC",

        # HTML & Structure
        "remove_html_tags": "Supprimer les balises HTML <tag>contenu</tag>",
        "remove_boilerplate": "Supprimer les phrases de template communes",

        # Transformation de casse
        "lowercase": "Convertir en minuscules",

        # Fusions structurelles
        "merge_dimensions": "Fusionner les motifs de dimensions (22 x 11 → 22x11)",
        "merge_units": "Fusionner les unités numériques (500 g → 500g)",
        "merge_durations": "Fusionner les durées (24 h → 24h)",
        "merge_age_ranges": "Fusionner les tranches d'âge (3-5 ans → 3_5ans)",
        "tag_years": "Étiqueter les années à 4 chiffres (1917 → year1917)",

        # Ponctuation
        "remove_punctuation": "Supprimer les signes de ponctuation isolés",

        # Filtrage de tokens
        "remove_stopwords": "Supprimer les mots vides français/anglais",
        "remove_single_letters": "Supprimer les caractères alphabétiques isolés",
        "remove_single_digits": "Supprimer les chiffres isolés",
        "remove_pure_punct_tokens": "Supprimer les tokens composés uniquement de ponctuation",
    }


def print_available_options():
    
    options = get_available_options()
    print("Options de nettoyage disponibles :")
    print("=" * 80)
    for option, description in options.items():
        print(f"  {option:25s} : {description}")
    print("=" * 80)

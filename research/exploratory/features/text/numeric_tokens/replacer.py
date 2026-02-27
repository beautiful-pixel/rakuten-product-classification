import re
from .dictionaries import LABELS_DICT, CONVERSION, UNIT_NORMALIZATION

def get_label(value: float, measure_type: str) -> str:
    """
    Associe une valeur numérique à un label discret selon un type de mesure.

    La discrétisation repose sur des seuils prédéfinis dans LABELS_DICT.

    Args:
        value (float): Valeur numérique exprimée dans l'unité de base.
        measure_type (str): Type de mesure (ex: 'volume', 'length', 'weight').

    Returns:
        str: Label discret correspondant à la valeur.
    """
    thresholds, labels = LABELS_DICT[measure_type]
    for i, label in enumerate(labels):
        if thresholds[i] <= value < thresholds[i + 1]:
            return label
    return measure_type + "inconnu"


def convert_to_base_unit(value: float, unit: str) -> float:
    """
    Convertit une valeur exprimée dans une unité donnée vers l'unité de base.

    Args:
        value (float): Valeur numérique.
        unit (str): Unité associée à la valeur.

    Returns:
        float: Valeur convertie dans l'unité de base.
    """
    if unit in CONVERSION:
        value = value * CONVERSION[unit]
    return value


def to_float(value):
    """
    retourne value sous forme de float
    """
    value = value.replace(',','.')
    value = float(value)
    return value

def multiply_values(values):
    """
    Multiplie une liste de valeurs numériques.

    Utilisé notamment pour le calcul de volumes à partir de dimensions.

    Args:
        values (list[float]): Liste de valeurs.

    Returns:
        float: Produit des valeurs.
    """
    computed_value = 1
    for v in values:
        computed_value = computed_value * v
    return computed_value

def decor_labels(labels):
    """
    Encapsule des labels sous forme de tokens textuels.

    Les labels sont entourés de crochets et les espaces sont remplacés
    par des underscores afin de former des tokens exploitables par les modèles.

    Args:
        labels (list[str]): Liste de labels à encapsuler.

    Returns:
        str: Représentation textuelle des tokens.
    """
    decored_labels = ["["+label.replace(' ','_')+"]" for label in labels]
    if len(decored_labels) > 1:
        txt = f"entre {decored_labels[0]} et {decored_labels[1]}"
    else:
        txt = decored_labels[0]
    return txt

def normalize_unit(unit: str) -> str:
    u = unit.lower()
    for pattern, normalized in UNIT_NORMALIZATION.items():
        if re.fullmatch(pattern, u):
            return normalized
    return u

def replace_numeric(match, measure_type, unit_pos=None, int_pos=None) -> str:
    """
    Remplace une expression numérique détectée par un token sémantique.

    Cette fonction :
    - extrait les valeurs numériques capturées par une expression régulière,
    - normalise et convertit les unités si nécessaire,
    - discrétise la valeur obtenue selon le type de mesure,
    - retourne un ou plusieurs tokens textuels.

    Args:
        match (re.Match): Correspondance regex contenant les groupes capturés.
        measure_type (str): Type de mesure (ex: 'volume', 'length', 'age').
        unit_pos (int, optional): Position du groupe correspondant à l'unité.
        int_pos (int, optional): Position du premier élément d'un intervalle.

    Returns:
        str: Token(s) textuel(s) remplaçant l'expression numérique.
    """
    if measure_type in LABELS_DICT:
        values = list(match.groups())
        if unit_pos is not None:
            unit = values.pop(unit_pos)
            unit = normalize_unit(unit)
            values = [convert_to_base_unit(to_float(v), unit) for v in values if v is not None]
        else:
            values = [to_float(v) for v in values if v is not None]
        if int_pos is not None:
            labels = [get_label(v, measure_type) for v in values]
        else:
            value = multiply_values(values)
            labels = [get_label(value, measure_type)]
    else:
        labels = [measure_type]
    txt = decor_labels(labels)
    return txt


REG_DIST = r"(mm|millim[eèé]tres?|cm|centim[eèé]tres?|m|m[eèé]tres?)"
REG_VOL = r"([mcd]?m3|[mcd]?l|millilitres?|centilitres?|litres?)"
REG_WEIGHT = r"(g|grammes?|kg|kilogrammes?|kilo|tonnes?)"

pattern_volume_m = rf"\b(\d+[.,]?\d*)\s*[xX*]\s*(\d+[.,]?\d*)\s*[xX*]\s*(\d+[.,]?\d*)\s*{REG_DIST}\b"
pattern_volume = rf"\b(\d+[.,]?\d*)\s*{REG_VOL}\b"
pattern_surface_m = rf"\b(\d+[.,]?\d*)\s*[xX*]\s*(\d+[.,]?\d*)\s*{REG_DIST}\b"
pattern_surface = r"\b(\d+[.,]?\d*)\s*([mcd]?m2)\b"
pattern_length = rf"\b(?:(\d+[.,]?\d*)\s*-\s*)?(\d+[.,]?\d*)\s*{REG_DIST}\b"
pattern_weight = rf"\b(\d+[.,]?\d*)\s*{REG_WEIGHT}\b"
pattern_price = r"\b(\d+[.,]?\d*)\s*(€|eur|euros?)\b"
pattern_age = r"\b(?:(\d+)\s*-\s*)?(\d+)\s*(mois|ans?)\b"
pattern_memory = r"\b(\d+)\s*([gmt][ob])\b"
pattern_date = r"\b(?:\d{2}/\d{2}/|\d{2}/)?((?:18|19|20)\d{2})\b"
pattern_numero = r"\b(num[eé]ro|num|n°|n.?|no.?)\s*(\d+)\b"
pattern_power = r"\b(\d+[.,]?\d*)\s*(cv|w|watts?)\b"
pattern_temperature = r"\b(\d+[.,]?\d*)\s*(°[ck]?|degr[eé]s?)\b"
pattern_energy = r"\b(\d+[.,]?\d*)\s*(kw|kilowatts?)\b"
pattern_capacity = r"\b(\d+[.,]?\d*)\s*(m?ah)\b"
pattern_tension = r"\b(\d+[.,]?\d*)\s*(v|volts?)\b"
pattern_card = r"\b(\d+)\s*(cartes?)\b"
pattern_piece = r"\b(\d+)\s*(pi[eè]ces?|pcs)\b"
pattern_integer = r"\b(\d+)\b"
pattern_float = r"\b(\d+[.,]?\d*)\b"

PATTERNS = [
    # pattern prix ???
    ("volume", pattern_volume_m, {"unit_pos": -1}),
    ("volume", pattern_volume, {"unit_pos": -1}),
    ("surface", pattern_surface_m, {"unit_pos": -1}),
    ("surface", pattern_surface, {"unit_pos": -1}),
    ("length", pattern_length, {"unit_pos": -1, "int_pos": 0}),
    ("weight", pattern_weight, {"unit_pos": -1}),
    ("price", pattern_price, {"unit_pos": -1}),
    ("age", pattern_age, {"unit_pos": -1, "int_pos": 0}),
    ("memory", pattern_memory, {"unit_pos": -1}),
    ("date", pattern_date, {}),
    ("numero", pattern_numero, {"unit_pos": 0}),
    ("power", pattern_power, {"unit_pos": -1}),
    ("energy", pattern_energy, {"unit_pos": -1}),
    ("capacity", pattern_capacity, {"unit_pos": -1}),
    ("temperature", pattern_temperature, {"unit_pos": -1}),
    ("tension", pattern_tension, {"unit_pos": -1}),
    ("card", pattern_card, {"unit_pos": -1}),
    ("piece", pattern_piece, {"unit_pos": -1}),
    ("integer", pattern_integer, {}),
    ("float", pattern_float, {}),
]

NUMERIC_GROUPS = {
    "light": {
        "volume", "surface", "length", "weight",
        "memory", "age", "card", "piece",
        "power", "energy", "capacity", "temperature", "tension"
    },
    "full": {
        "volume", "surface", "length", "weight",
        "memory", "age", "card", "piece",
        "power", "energy", "capacity", "temperature", "tension",
        "date", "numero", "integer", "float", "price"
    },
    "phys": {
        "length",
        "surface",
        "volume",
        "weight",
    },
}


def replace_numeric_expressions(
    txt: str,
    mode: str = "light",
    enabled_measures: set[str] | None = None,
) -> str:
    """
    Remplace les expressions numériques par des tokens sémantiques,
    selon une stratégie paramétrable.

    Args:
        txt (str): Texte brut.
        mode (str): Stratégie de normalisation numérique
            ("light" ou "full" ou "phys").

    Returns:
        str: Texte avec tokens numériques.
    """
    if enabled_measures is None:
        enabled_measures = NUMERIC_GROUPS[mode]
        
    txt = re.sub(r"²", "2", txt)
    txt = re.sub(r"³", "3", txt)

    enabled_measures = NUMERIC_GROUPS[mode]

    for measure, pattern, params in PATTERNS:
        if measure not in enabled_measures:
            continue

        txt = re.sub(
            pattern,
            lambda m: replace_numeric(m, measure, **params),
            txt,
            flags=re.IGNORECASE
        )

    return txt


USED_MEASURES = {m for m, _, _ in PATTERNS}

def get_all_numeric_tokens():
    """
    Retourne l'ensemble de tous les tokens numériques possibles.

    Returns:
        list[str]: Liste de tokens.
    """
    tokens = []
    for measure in USED_MEASURES:
        if measure in LABELS_DICT:
            tokens += [decor_labels([l]) for l in LABELS_DICT[measure][1]]
        else:
            tokens.append(decor_labels([measure]))
    return tokens

def get_numeric_tokens(measure_types=None):
    """
    Retourne la liste des tokens numériques associés aux types de mesures demandés.

    Args:
        measure_types (list[str] | None): Liste des types de mesures à considérer
            (par exemple `["length", "weight", "date"]`).  
            Si `None`, l'ensemble des tokens numériques possibles est retourné.

    Returns:
        list[str]: Liste des tokens numériques formatés, prêts à être utilisés
        dans un pipeline de traitement du texte ou pour l'analyse exploratoire.
    """

    tokens = []
    if measure_types is None:
        tokens = get_all_numeric_tokens()
    else:
        for measure in measure_types:
            if measure in LABELS_DICT:
                tokens += [decor_labels([l]) for l in LABELS_DICT[measure][1]]
            elif measure in USED_MEASURES:
                tokens.append(decor_labels([measure]))
    return tokens
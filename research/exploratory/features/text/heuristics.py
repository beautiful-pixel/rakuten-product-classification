import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re
import pandas as pd


# 1. Definir les mots clés par catégorie

KEYWORD_DICT = {
    "Animaux": [ "chien", "chat", "animal", "compagnie", "collier"],
    "Bureau & Papeterie": ["verso", "cahier", "encre", "papier", "recto", "a5"],
    "Épicerie": ["epices", "arôme", "chocolat", "sucre", "sachet", "capsule"],
    "Puériculture": ["langer", "bavoir", "assiette", "siege", "tétine", "poussette"],
    "Vêtement Bébé & Loisirs": ["bébé", "chaussettes", "paire", "longueur", "filles","garçons"],
    "Figurines": ["figurine", "gundam", "statuette", "officiel", "marvel", "funko"],
    "Jeux de cartes":["mtg", "oh", "rare", "vf", "carte", "magic"],
    "Jeux de rôle & Figurines": ["halloween", "figurine", "warhammer", "prince", "masque"],
    "Bricolage & Outillage": ["arrosage", "tondeuse", "aspirateur", "appareils", "outil", "coupe", "bâche"],
    "Décoration & Équipement Jardin": ["bois", "jardin", "résistant", "tente", "parasol", "aluminium"],
    "Piscine & Accessoires": ["piscine", "filtration", "pompe", "dimensions","eau", "ronde"],
    "Accessoires & Périphériques":["nintendo", "manette", "protection", "ps4", "silicone", "câble"],
    "Consoles": ["console", "oui", "jeu", "écran", "portable", "marque", "jeux"],
    "Jeux PC en Téléchargement":["windows", "jeu", "directx", "plus", "téléchargement", "disque", "édition"],
    "Jeux Vidéo Modernes": ["duty","jeux", "manettes", "ps3", "xbox", "kinect"],
    "Rétro Gaming": ["japonais", "import", "langue", "titres", "sous", "français"],
    "Jeux éducatifs": ["joue", "cartes", "enfants", "éducatif", "bois", "jouer"],
    "Jouets & Figurines": ["doudou", "enfants", "cadeau", "peluche", "jouet", "puzzle"],
    "Loisirs & Plein air": ["camping", "pêche", "stress", "stream", "bracelet", "trampoline"],
    "Modélisme & Drones": ["drone", "générique", "dji", "avion", "batterie", "cámera", "one"],
    "Littérature": ["monde", "ouvrage", "siècle", "roman", "livre", "histoire", "tome"],
    "Livres spécialisés": ["guide", "édition", "histoire", "art", "collection"],
    "Presse & Magazines": ["journal", "france", "illustre", "magazine", "presse", "revue"],
    "Séries & Encyclopédies":[ "lot", "livres", "tomes", "volumes", "tome", "revues"],
    "Décoration & Lumières": ["led", "noël", "lumière", "lampe", "décoration", "couleur"],
    "Textiles d'intérieur": ["oreiller", "taie", "coussin", "couverture", "canapé", "cotton"],
     "Équipement Maison":["matelas", "assise", "bois", "table", "hauteur", "mousse"]
}



# # txt doit etre un pd.Series
# def count_kw(txt):
#     data = {}
#     for cat, mots in keyword_dict.items():
#         pattern = '|'.join(mots)  # expression régulière
#         data[cat + "_keywords"] = txt.str.count(pattern)
#     return pd.DataFrame(data)


# params = {
#     'fix_encoding':True, 'unescape_html':True,
#     'normalize_unicode':True, 'remove_html_tags':True,
#     'lowercase':True
# }

# def preprocess_kw(df):
#     txt = df["designation"].fillna("") + " " +df["description"].fillna("")
#     txt = txt.apply(lambda x : clean_text(x, **params))
#     return count_kw(txt)



class KeywordFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, keyword_dict=None):
        self.keyword_dict = keyword_dict if keyword_dict else KEYWORD_DICT

    def fit(self, X, y=None):
        self.feature_names_ = [
            f"{cat}_keywords" for cat in self.keyword_dict.keys()
        ]
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : pd.Series or array-like of shape (n_samples,)
            Colonne de texte unique.
        """
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        
        txt = X.fillna("")

        data = {}
        for cat, mots in self.keyword_dict.items():
            pattern = "|".join(mots)
            data[f"{cat}_keywords"] = txt.str.count(pattern)

        return pd.DataFrame(data, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


UNIT_PATTERNS = {
    # Dimensions
    "cm":   r"\b\d+\s*(cm|centimetre?s?|centimètre?s?)\b",
    "mm":   r"\b\d+\s*(mm|millimetre?s?|millimètre?s?)\b",
    "m":    r"\b\d+\s*(m|metre?s?|mètre?s?)\b",
    # Poids
    "kg":   r"\b\d+\s*(kg|kilo|kilogramme?s?)\b",
    "g":    r"\b\d+\s*(g|gramme?s?)\b",
    # Volume
    "ml":   r"\b\d+\s*(ml|millilitres?|millilitre?)\b",
    "l":    r"\b\d+\s*(l|litres?|litre?)\b",
    "cl":   r"\b\d+\s*cl\b",  
    # Dimensions suivies d'une unité
    "x_dim": r"\b\d+\s*(x|×)\s*\d+(\s*(cm|mm|m))?\b",
    # Âge
    "age_ans":  r"\b\d+\s*ans\b",
    "age_mois": r"\b\d+\s*mois\b",
    # Tranches d'âge: 
    "age_range_ans":  r"\b\d+\s*[-–/]\s*\d+\s*ans\b",
    "age_range_mois": r"\b\d+\s*[-–/]\s*\d+\s*mois\b",
    # Age minimum 
    "age_min_ans": r"\b(?:à\s*partir\s*de\s*)?\d+\s*ans\s*(?:et\s*plus|\+|plus)?\b",
    # Pointure 
    "pointure": r"\bpointures?\s*:?(\s*(eu|eur|fr))?\s*"
                r"\d{2}(?:[.,]\d)?(\s*[-/]\s*\d{2}(?:[.,]\d)?)?\b",
    # Pouces (écrans)
    "inch": r'\b\d+\s*(\"|pouces?|po)\b',
    # Temps 
    "time_heures": r"\b\d+\s*(?:h|heures?)\b",       
    "time_jours":  r"\b\d+\s*jours?\b",             
    # Stockage / mémoire
    "go":   r"\b\d+\s*(go|giga-?octets?)\b",        
    "gb":   r"\b\d+\s*(gb|gigabytes?)\b",          
    "mo":   r"\b\d+\s*(mo|méga-?octets?|mega-?octets?)\b", 
    "mb":   r"\b\d+\s*(mb|megabytes?)\b",         
    "ko":   r"\b\d+\s*(ko|kilo-?octets?)\b",      
    "kb":   r"\b\d+\s*(kb|kilobytes?)\b",
    # RAM / VRAM3
    "ram":  r"\b\d+\s*(go|gb|mo|mb)\s*ram\b",      
    "vram": r"\b\d+\s*(go|gb|mo|mb)\s*vram\b",      
    # Fréquence CPU / GPU
    "ghz":  r"\b\d+(?:\.\d+)?\s*ghz\b",           
    "mhz":  r"\b\d+(?:\.\d+)?\s*mhz\b",
    # Architecture (32-bit / 64-bit)
    "bits": r"\b(32|64)\s*[- ]?(?:bit|bits)\b",     
    # Débit réseau
    "kbps": r"\b\d+\s*kbps\b",                     
    "mbps": r"\b\d+\s*mbps\b",
    # Framerate
    "fps":  r"\b\d+\s*fps\b",                       
}

def detect_any_unit(text, list_of_regex):
    """
    Retourne 1 si le texte contient au moins une unité (parmi la liste de regex),
    sinon 0.
    """
    text = str(text)

    for regex_pattern in list_of_regex:
        if regex_pattern.search(text):
            return 1
    return 0

def count_digits(text):
    """
    Compte le nombre de chiffres (0-9) dans une chaîne de caractères.
    """
    compteur = 0
    for caractere in text:
        if caractere.isdigit():
            compteur = compteur + 1
    return compteur

# def preprocess_unit(df):
#     txt = df["designation"].fillna("") + " " +df["description"].fillna("")
#     txt = txt.apply(lambda x : clean_text(x, **params))
#     data = {}
#     data["nb_digits_text"] = txt.apply(count_digits)
#     data["contains_numerotation"] = np.where(
#         txt.str.contains(r"n° ?([0-9])+"), 1, 0
#     )
#     data["has_any_unit"] = txt.apply(
#         lambda txt: detect_any_unit(txt, compiled_patterns)
#     )
#     return pd.DataFrame(data)

class UnitFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.feature_names_ = [
            "nb_digits_text",
            "contains_numerotation",
            "has_any_unit",
        ]
        return self

    def transform(self, X):
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        txt = X.fillna("")
        compiled_patterns = []
        for pattern in UNIT_PATTERNS.values():
            regex_obj = re.compile(pattern, flags=re.IGNORECASE)
            compiled_patterns.append(regex_obj)

        data = {
            "nb_digits_text": txt.apply(count_digits),
            "contains_numerotation": np.where(
                txt.str.contains(r"n° ?(?:[0-9])+"), 1, 0
            ),
            "has_any_unit": txt.apply(
                lambda t: detect_any_unit(t, compiled_patterns)
            ),
        }

        return pd.DataFrame(data, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


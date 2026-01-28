import numpy as np


# =========================
# Dictionnaire des labels
# =========================

LABELS_DICT = {
    'volume' : (
        # volume moyen de 5 litres à 1.5 m**3
        [0, 5e3, 1.5e6, np.float32('inf')],
        ['petit volume', 'volume moyen', 'grand volume']
    ),
    'surface' : (
        [0, 1600, 2e4, np.float32('inf')],
        ['petite surface', 'surface moyenne', 'grande surface']
    ),
    'length' : (
        [0, 10, 40, 350, np.float32('inf')],
        ['petite longueur', 'longueur moyenne', 'grande longueur', 'très grande longueur']
    ),
    'weight' : (
        [0, 1.5, 20, np.float32('inf')],
        ['poids leger', 'poids moyen', 'poids lourd']
    ),
    'price' : (
        [0, 20, 100, np.float32('inf')],
        ['prix bas', 'prix moyen', 'prix eleve']
    ),
    'age' : (
        [0, 3, 15, np.float32('inf')],
        ['age bébé', 'age enfant', 'age adulte']
    ),
    'memory' : (
        [0, 8, 64, np.float32('inf')],
        ['petite mémoire', 'mémoire moyenne', 'grande mémoire']
    ),
    'date' : (
        [1800, 1960, 2007, 2021, np.float32('inf')],
        ['date ancienne', 'date contemporaine', 'date récente', 'date future']
    ),
    'numero' : (
        [0, 20, np.float32('inf')],
        ['petit numero', 'grand numero']
    ),
    'card' : (
        [0, 5, np.float32('inf')],
        ['peu de cartes', 'beaucoup de cartes']
    ),
    'piece' : (
        [0, 3, np.float32('inf')],
        ['peu de pièces', 'beaucoup de pièces']
    ),
    'integer' : (
        [0, 1, 2, 5, 20, 100, np.float32('inf')],
        ['zéro', 'un', 'petit nombre', 'nombre moyen', 'grand nombre', 'très grand nombre']
    ),
    'float' : (
        [0, 1, 10, 100, np.float32('inf')],
        ['petite quantité', 'quantité moyenne', 'grande quantité', 'très grande quantité']
    )
}


# LABELS_DICT = {
#     'volume' : (
#         # volume moyen de 5 litres à 1.5 m**3
#         [0, 5e3, 1.5e6, np.float32('inf')],
#         ['petit volume', 'volume moyen', 'grand volume']
#     ),
#     'surface' : (
#         [0, 1600, 2e4, np.float32('inf')],
#         ['petite surface', 'surface moyenne', 'grande surface']
#     ),
#     'length' : (
#         [0, 10, 40, 350, np.float32('inf')],
#         ['petite longueur', 'longueur moyenne', 'grande longueur', 'très grande longueur']
# ),
#     'weight' : (
#         [0, 1.5, 20, np.float32('inf')],
#         ['poids leger', 'poids moyen', 'poids lourd']
#     ),
#     'age' : (
#         [0, 3, 15, np.float32('inf')],
#         ['age bébé', 'age enfant', 'age adulte']
#     ),
#     'memory' : (
#         [0, 10, np.float32('inf')],
#         ['petite mémoire', 'grande mémoire']
#     ),
#     'date' : (
#         [1800, 1960, 2007, 2021, np.float32('inf')],
#         ['date ancienne', 'date contemporaine', 'date récente', 'date future']
#     ),
#     'numero' : (
#         [0, 20, np.float32('inf')],
#         ['petit numero', 'grand numero']
#     ),
#     'card' : (
#         [0, 5, np.float32('inf')],
#         ['peu de cartes', 'beaucoup de cartes']
#     ),
#     'piece' : (
#         [0, 3, np.float32('inf')],
#         ['peu de pièces', 'beaucoup de pièces']
#     ),
#     'integer' : (
#         [0, 1, 2, 5, 20, 100, np.float32('inf')],
#         ['zéro', 'un', 'petit nombre', 'nombre moyen', 'grand nombre', 'très grand nombre']
#     ),
#     'float' : (
#         [0, 1, 10, 100, np.float32('inf')],
#         ['petite quantité', 'quantité moyenne', 'grande quantité', 'très grande quantité']
#     )
# }

# =========================
# Normalisation des unités
# =========================

UNIT_NORMALIZATION = {
    r"m[èeé]tres?" : "m", r"centim[eèé]tres?" : "cm", r"millim[eèé]tres?" : "mm",
    r"grammes?" : "g", r"kilos?|kilogrammes?" : "kg", r"tonnes?" : "tonne",
    r"litres?" : "l", r"centilitres?" : "cl", r"millilitres?" : "ml",
    r"ans?" : "ans",
}

# =========================
# Conversion d'unités
# =========================

CONVERSION = {
    'mm' : 0.1, 'cm' : 1, 'dm' : 10, 'm' : 100,
    'mm2' : 0.01, 'cm2' : 1, 'dm2' : 100, 'm2' : 10**4,
    'mm3' : 10**-3, 'cm3' : 1, 'dm3' : 1000, 'm3' : 10**6,
    'ml' : 1, 'cl' : 10, 'dl' : 100, 'l' : 1000,
    'g' : 1e-3, 'kg' : 1, 'tonne' : 1e3,
    'mois' : 1/12, 'ans' : 1,
    'mo' : 10**-3, 'go' : 1, 'to' : 10**3,
}

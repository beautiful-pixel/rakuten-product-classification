from typing import Dict, Optional, List
import pandas as pd


# Full category names
CATEGORY_NAMES = {
    10: "Livres techniques, éducatifs, artistiques ou spirituels",
    2705: "Romans, récits et littérature",
    2280: "Journaux, magazines et revues",
    2403: "Séries & encyclopédies",
    40: "Rétro Gaming",
    50: "Accessoires & Périphériques de Jeux Vidéo",
    60: "Consoles",
    2462: "Jeux Vidéo Modernes",
    2905: "Jeux PC en Téléchargement",
    1140: "Figurine",
    1160: "Jeu de carte à collectionner",
    1180: "Jeux de rôle & figurines",
    1280: "Jouets, Figurines et Poupées",
    1281: "Jeux éducatifs & Créatifs",
    1300: "Modélisme & Drones",
    1301: "Bébé, Jeux & Loisirs",
    1302: "Sport, Loisirs & Plein Air",
    1320: "Bébé & Puériculture",
    1560: "Équipement de la maison & décoration",
    1920: "Textiles d'intérieur",
    2060: "Décoration & Éclairage",
    2582: "Jardinage, déco & extérieur",
    2583: "Piscine & équipement de piscine",
    2585: "Jardin, Bricolage & Outillage",
    1940: "Alimentation & Épicerie",
    2220: "Animaux & Accessoires",
    2522: "Fournitures de bureau & papeterie",
}


# Short category names for display (max ~30 chars)
CATEGORY_SHORT_NAMES = {
    10: "Livres techniques",
    2705: "Romans & littérature",
    2280: "Journaux & magazines",
    2403: "Séries & encyclopédies",
    40: "Rétro Gaming",
    50: "Accessoires JV",
    60: "Consoles",
    2462: "Jeux Vidéo",
    2905: "Jeux PC",
    1140: "Figurine",
    1160: "Cartes à collectionner",
    1180: "Jeux de rôle",
    1280: "Jouets & Figurines",
    1281: "Jeux éducatifs",
    1300: "Modélisme & Drones",
    1301: "Bébé & Jeux",
    1302: "Sport & Loisirs",
    1320: "Bébé & Puériculture",
    1560: "Équipement maison",
    1920: "Textiles",
    2060: "Déco & Éclairage",
    2582: "Jardinage & déco",
    2583: "Piscine",
    2585: "Jardin & Bricolage",
    1940: "Alimentation",
    2220: "Animaux",
    2522: "Fournitures bureau",
}


# Category groups for analysis
CATEGORY_GROUPS = {
    "Livres & Médias": [10, 2705, 2280, 2403],
    "Jeux Vidéo": [40, 50, 60, 2462, 2905],
    "Jouets & Loisirs": [1140, 1160, 1180, 1280, 1281, 1300],
    "Bébé & Enfant": [1301, 1320],
    "Maison & Jardin": [1560, 1920, 2060, 2582, 2583, 2585],
    "Sport & Vie": [1302, 1940, 2220, 2522],
}


def get_category_name(code: int, short: bool = False) -> str:
    """
    Return the human-readable category name for a given category code.

    Args:
        code (int): Category code.
        short (bool, optional): If True, return the short display name.

    Returns:
        str: Category name or the code as string if not found.
    """
    mapping = CATEGORY_SHORT_NAMES if short else CATEGORY_NAMES
    return mapping.get(code, str(code))


def get_all_categories(short: bool = False) -> Dict[int, str]:
    """
    Return a mapping of all category codes to their names.

    Args:
        short (bool, optional): If True, return short display names.

    Returns:
        Dict[int, str]: Mapping from category code to category name.
    """
    return CATEGORY_SHORT_NAMES.copy() if short else CATEGORY_NAMES.copy()


def get_category_codes() -> List[int]:
    """
    Return all available category codes sorted in ascending order.

    Returns:
        List[int]: Sorted category codes.
    """
    return sorted(CATEGORY_NAMES.keys())


def get_category_group(code: int) -> Optional[str]:
    """
    Return the category group associated with a category code.

    Args:
        code (int): Category code.

    Returns:
        Optional[str]: Group name if the code belongs to a group,
        otherwise None.
    """
    for group_name, codes in CATEGORY_GROUPS.items():
        if code in codes:
            return group_name
    return None


def format_category_label(
    code: int,
    show_code: bool = True,
    short: bool = False,
    max_length: Optional[int] = None
) -> str:
    """
    Format a category label for display.

    Args:
        code (int): Category code.
        show_code (bool, optional): If True, prepend the code to the label.
        short (bool, optional): If True, use the short category name.
        max_length (Optional[int], optional): Maximum label length.
            If exceeded, the label is truncated with ellipsis.

    Returns:
        str: Formatted category label.
    """
    name = get_category_name(code, short=short)

    if show_code:
        label = f"{code}: {name}"
    else:
        label = name

    if max_length and len(label) > max_length:
        label = label[:max_length-3] + "..."

    return label


def map_categories_in_dataframe(
    df: pd.DataFrame,
    code_column: str = "prdtypecode",
    target_column: str = "category_name",
    short: bool = False
) -> pd.DataFrame:
    """
    Map category codes in a DataFrame to human-readable names.

    Args:
        df (pd.DataFrame): Input DataFrame.
        code_column (str, optional): Column containing category codes.
        target_column (str, optional): Name of the output column.
        short (bool, optional): If True, use short category names.

    Returns:
        pd.DataFrame: Copy of the DataFrame with an added category name column.
    """
    df = df.copy()
    mapping = CATEGORY_SHORT_NAMES if short else CATEGORY_NAMES
    df[target_column] = df[code_column].map(mapping).fillna(df[code_column].astype(str))
    return df


def get_category_distribution(
    codes: pd.Series,
    short: bool = True,
    sort_by: str = "count"
) -> pd.DataFrame:
    """
    Compute category distribution statistics.

    Args:
        codes (pd.Series): Series of category codes.
        short (bool, optional): If True, use short category names.
        sort_by (str, optional): Sorting criterion ("count", "code", "name").

    Returns:
        pd.DataFrame: Category distribution with counts and percentages.
    """
    counts = codes.value_counts()
    total = len(codes)

    result = pd.DataFrame({
        'code': counts.index,
        'name': [get_category_name(c, short=short) for c in counts.index],
        'count': counts.values,
        'percentage': (counts.values / total * 100).round(2)
    })

    if sort_by == 'count':
        result = result.sort_values('count', ascending=False)
    elif sort_by == 'code':
        result = result.sort_values('code')
    elif sort_by == 'name':
        result = result.sort_values('name')

    return result.reset_index(drop=True)


def validate_category_code(code: int) -> bool:
    return code in CATEGORY_NAMES


def print_category_summary():
    
    print("=" * 80)
    print("RAKUTEN CATEGORY MAPPING")
    print("=" * 80)
    print(f"Total categories: {len(CATEGORY_NAMES)}")
    print()

    for group_name, codes in CATEGORY_GROUPS.items():
        print(f"\n{group_name} ({len(codes)} categories):")
        print("-" * 60)
        for code in sorted(codes):
            name = get_category_name(code, short=False)
            short_name = get_category_name(code, short=True)
            print(f"  {code:4d}: {short_name:30s} ({name})")

    print("\n" + "=" * 80)
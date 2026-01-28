from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import yaml
from PIL import Image
from data.categories import CATEGORY_NAMES, CATEGORY_GROUPS
from inference.load_model import load_model
from inference.predictor import Predictor


# Model & Inference Setup

def load_config():
    with open(Path(__file__).parent / "config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
USE_API = config["inference"]["use_api"]

if not USE_API:
    @st.cache_resource(show_spinner=True)
    def load_demo_model():
        return load_model(
            repo_id=config["model"]["repo_id"]
        )
    model = load_demo_model()
else:
    model = None

predictor = Predictor(config=config, model=model)

# Page Configuration

st.set_page_config(
    page_title="Rakuten – Classification Multimodale de Produits",
    layout="wide"
)


# UI Theme & Constants

COLORS = {
    "primary": "#34506D",
    "secondary": "#f2f2f2",
    "accent": "#d9822b"
}


# Custom CSS (Streamlit Overrides)

st.markdown(f"""
<style>
/* Conteneur principal */
.block-container {{
    max-width: 1400px;
    padding-left: 2rem;
    padding-right: 2rem;
}}
            
/* Titres */
.stMarkdown h2, .stMarkdown h3 {{
    color: {COLORS['primary']};
}}
/* Puces catégories */
.category-pill {{
    display: inline-block;
    background-color: {COLORS['secondary']};
    border-left: 4px solid {COLORS['accent']};
    padding: 4px 8px;
    margin: 4px 4px 4px 0;
    border-radius: 6px;
    font-size: 0.9em;
}}
/* Boutons */
.stButton>button {{
    background-color: {COLORS['accent']};
    color: white;
}}
/* Résultat mis en valeur */
.result-box {{
    background-color: {COLORS['secondary']};
    border-left: 6px solid {COLORS['accent']};
    padding: 8px;
    border-radius: 8px;
    margin-bottom: 24px;
}}
</style>
""", unsafe_allow_html=True)


# User Interface

tab1, tab2, tab3 = st.tabs([
    "Le projet",
    "Catégories couvertes",
    "Tester le modèle"
])

with tab1:

    st.markdown("""
    ## Classification automatique de produits e-commerce
    Ce projet a pour objectif de **prédire automatiquement la catégorie d’un produit**,
    parmi **27 catégories**, à partir de sa **désignation**, de sa **description**
    et de son **image**, en combinant des modèles spécialisés au sein d’un
    **pipeline multimodal**.
    """)

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:

        st.markdown("""
        <div class="result-box">
            <h4>Contexte</h4>
            <p>
                Ce travail a été réalisé dans le cadre du challenge
                <a href="https://challengedata.ens.fr/challenges/35" target="_blank">
                Rakuten France Multimodal Product Classification
                </a>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)


        st.markdown("""
        <div class="result-box">
            <h4>Approche</h4>
            <ul>
                <li>Modèles spécialisés pour le texte et l’image</li>
                <li>Fusion tardive des prédictions via un méta-modèle</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(
            f"""
            <div class="result-box">
                <h4>Résultats</h4>
                <p>
                Score F1-pondéré sur le jeu de test du challenge<br>
                <h2>92.7 %</h2>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

with tab2:
    st.markdown("## Catégories prédites par le modèle")

    for group_name, codes in CATEGORY_GROUPS.items():
        st.markdown(f"### {group_name}")

        pills = "".join(
            f"<span class='category-pill'>{CATEGORY_NAMES[code]}</span>"
            for code in codes
        )

        st.markdown(pills, unsafe_allow_html=True)

with tab3:

    st.markdown("## Tester le modèle")

    col_inputs, col_results = st.columns([1.1, 0.9], gap="large")

    with col_inputs:


        st.markdown("### Entrées")

        designation = st.text_input(
            "Désignation du produit *",
            placeholder="Ex : Figurine Dragon Ball Z – Goku Super Saiyan"
        )

        description = st.text_area(
            "Description du produit (optionnelle)",
            placeholder="Figurine en PVC, hauteur 18 cm...",
        )

        st.markdown("Image du produit *")

        img_col_left, img_col_right = st.columns([1.2, 0.8], gap="large")

        with img_col_left:

            image_mode = st.radio(
                "Mode",
                ["Uploader ma propre image", "Utiliser une image de démonstration"],
                label_visibility="collapsed"
            )

            demo_dir = Path(__file__).parent / "demo_images"
            demo_images = sorted(demo_dir.glob("*.jpg"))

            selected_image = None

            if image_mode == "Utiliser une image de démonstration":
                demo_choice = st.selectbox(
                    "Choisir une image",
                    demo_images,
                    format_func=lambda p: p.stem.replace("_", " ").title()
                )
                selected_image = Image.open(demo_choice).convert("RGB")

            else:
                uploaded = st.file_uploader(
                    "Uploader une image",
                    type=["jpg", "jpeg", "png"]
                )
                if uploaded:
                    selected_image = Image.open(uploaded).convert("RGB")

            with img_col_right:

                if selected_image:
                    st.image(
                        selected_image,
                        caption="Aperçu",
                        width=180
                    )

        st.markdown("---")
        run = st.button("Lancer la prédiction", use_container_width=True)

    if run:

        if not designation.strip():
            st.error("La désignation est obligatoire.")
            st.stop()

        if selected_image is None:
            st.error("Une image est obligatoire.")
            st.stop()

        with st.spinner("Prédiction en cours..."):
            result = predictor.predict_with_contributions(
                designation=designation,
                description=description,
                image=selected_image,
            )

        result["category_name"] = CATEGORY_NAMES[result["label_pred"]]
        st.session_state.result = result

    with col_results:

        st.markdown("### Résultat")

        if "result" in st.session_state:

            result = st.session_state.result

            st.markdown(
                f"""
                <div class="result-box">
                    <strong>Catégorie prédite</strong><br>
                    <code>{result["label_pred"]}</code> — <strong>{result["category_name"]}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Confiance globale (méta-modèle)",
                f"{result['P_final']:.2f}"
            )

            st.markdown("**Contributions par modalité**")

            st.progress(float(result["P_text"]))
            st.caption(f"Texte : **{float(result['P_text']):.2f}**")

            st.progress(float(result["P_image"]))
            st.caption(f"Image : **{float(result['P_image']):.2f}**")
        else:
            st.info("Renseignez les informations et lancez une prédiction.")


    st.info(
        "Les données saisies (texte et image) **ne sont pas stockées** "
        "et sont utilisées uniquement pour cette prédiction."
    )
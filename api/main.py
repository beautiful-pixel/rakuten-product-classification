from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, UploadFile, File, Form
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import yaml
from PIL import Image
import io
from inference.load_model import load_model

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    app.state.model = load_model(
        repo_id=config["model"]["repo_id"]
    )
    yield


app = FastAPI(title="Rakuten Product Classification", lifespan=lifespan)


def build_input_dataframe(designation: str, description: str | None):
    return pd.DataFrame({
        "designation": [designation],
        "description": [description if description else np.nan],
    })

async def build_input_images(image: UploadFile):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return [pil_image]


@app.post("/predict")
async def predict(
    designation: str = Form(...),
    description: str | None = Form(None),
    image: UploadFile = File(...)
):
    texts_df = build_input_dataframe(designation, description)
    images = await build_input_images(image)
    y = app.state.model.predict(texts_df, images)
    return int(y[0])


@app.post("/predict_proba")
async def predict_proba(
    designation: str = Form(...),
    description: str | None = Form(None),
    image: UploadFile = File(...)
):
    texts_df = build_input_dataframe(designation, description)
    images = await build_input_images(image)
    probas = app.state.model.predict_proba(texts_df, images)
    return probas[0].tolist()


@app.post("/predict_labels")
async def predict_labels(
    designation: str = Form(...),
    description: str | None = Form(None),
    image: UploadFile = File(...)
):
    texts_df = build_input_dataframe(designation, description)
    images = await build_input_images(image)
    labels = app.state.model.predict_labels(texts_df, images)
    return labels[0]


@app.post("/predict_with_contributions")
async def predict_with_contributions(
    designation: str = Form(...),
    description: str | None = Form(None),
    image: UploadFile = File(...)
):
    texts_df = build_input_dataframe(designation, description)
    images = await build_input_images(image)
    df = app.state.model.predict_with_contributions(texts_df, images)
    return df.to_dict(orient="records")[0]

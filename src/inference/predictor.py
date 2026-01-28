import io
import requests
import pandas as pd
import numpy as np
from PIL import Image


class Predictor:
    def __init__(self, config, model=None):
        self.use_api = config["inference"]["use_api"]
        self.api_url = config["inference"].get("api_url")
        self.model = model

        if not self.use_api and self.model is None:
            raise ValueError("Local mode requires a loaded model.")

    def predict_with_contributions(
        self,
        designation: str,
        description: str | None,
        image: Image.Image,
    ):
        if self.use_api:
            return self._predict_via_api(designation, description, image)
        else:
            return self._predict_local(designation, description, image)

    def _predict_via_api(self, designation, description, image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        response = requests.post(
            self.api_url,
            data={
                "designation": designation,
                "description": description,
            },
            files={
                "image": ("image.png", buf, "image/png")
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def _predict_local(self, designation, description, image):
        df = pd.DataFrame([{
            "designation": designation,
            "description": description if description else np.nan,
        }])

        result = self.model.predict_with_contributions(
            texts=df,
            images=[image],
        )

        return result.to_dict(orient="records")[0]

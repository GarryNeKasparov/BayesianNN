import shutil
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pyro.infer import Predictive

from bayesiannn.models.Unet import Unet

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
UPLOADS_DIR = STATIC_DIR / "uploads"
OUTPUTS_DIR = STATIC_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "model_last.pt"
WEIGHTS_PATH = MODEL_DIR / "weights.pt"
GUIDE_PATH = MODEL_DIR / "guide.pt"

model: Unet = None
transform: nn.Module = None


def check_files_exist():
    required_files = [MODEL_PATH, WEIGHTS_PATH, GUIDE_PATH]
    missing_files = [str(file) for file in required_files if not file.exists()]
    if missing_files:
        raise FileNotFoundError(
            (
                "The following required files"
                f" are missing: {', '.join(missing_files)}"
            )
        )


def load_model(app: FastAPI):
    global model, transform
    check_files_exist()
    try:
        model = Unet(bayesian=True)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        guide = torch.load(GUIDE_PATH, map_location="cpu")
        model = Predictive(
            model, num_samples=10, guide=guide, return_sites=("obs", "_RETURN")
        )
        transform = A.Compose(
            [
                A.Resize(256, 256),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1]),
                ToTensorV2(),
            ]
        )
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e
    yield
    del model, transform
    print("Model unloaded.")


app = FastAPI(lifespan=load_model)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    try:
        return transform(image=image)["image"].unsqueeze(0)
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}") from e


def run_inference(model: Unet, image_tensor: torch.Tensor) -> np.ndarray:
    try:
        with torch.no_grad():
            output: torch.Tensor = model(image_tensor)
            predictions = output["obs"].numpy()
        return predictions
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {e}") from e


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Handle file upload and perform inference."""
    global model

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    file_path = UPLOADS_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(str(file_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    try:
        input_tensor = preprocess_image(image)
        predictions = run_inference(model, input_tensor)
        mean_prediction = predictions.mean(axis=0).squeeze()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    input_image_path = OUTPUTS_DIR / "input_image.png"
    mean_prediction_path = OUTPUTS_DIR / "mean_prediction.png"

    plot_image = input_tensor.squeeze(0).numpy().transpose(1, 2, 0) + 0.5
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(plot_image)
    plt.axis("off")
    fig.savefig(input_image_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(plot_image)
    ax.imshow(mean_prediction, alpha=0.5, cmap="viridis")
    plt.axis("off")
    fig.savefig(mean_prediction_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    individual_prediction_paths = []
    for i, prediction in enumerate(predictions):
        pred_path = OUTPUTS_DIR / f"prediction_{i}.png"
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(plot_image)
        ax.imshow(prediction.squeeze(), alpha=0.5, cmap="viridis")
        plt.axis("off")
        plt.savefig(pred_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        individual_prediction_paths.append(
            f"/static/outputs/prediction_{i}.png"
        )

    return RedirectResponse(
        url=(
            "/results?input_image=/static/outputs/input_image.png&"
            "mean_prediction="
            "/static/outputs/mean_prediction.png&predictions="
            f"{','.join(individual_prediction_paths)}"
        ),
        status_code=303,
    )


@app.get("/results", response_class=HTMLResponse)
async def show_results(
    request: Request,
    input_image: str,
    mean_prediction: str,
    predictions: str,
):
    individual_predictions = predictions.split(",")
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "input_image": input_image,
            "mean_prediction": mean_prediction,
            "individual_predictions": individual_predictions,
        },
    )

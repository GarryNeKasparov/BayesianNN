import os
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
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pyro.infer import Predictive

from bayesiannn.models.Unet import Unet

model: Unet = None
transform: nn.Module = None
MODEL_PATH = "../models/mymodel.pt"


def load_model(app: FastAPI):
    global model, transform
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    try:
        model = Unet(bayesian=True)
        model.load_state_dict(torch.load("../models/weights.pt"))
        guide = torch.load("../models/guide_weights.pt")
        # params = torch.load(MODEL_PATH)
        # model.load_state_dict(torch.load("../models/raw.pt"))
        # guide = params["guide"]
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
        print(f"Error loading model: {str(e)}")
    yield

    del model, transform
    print("Model unloaded.")


app = FastAPI(lifespan=load_model)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def preprocess_image(image: Image.Image):
    return transform(image=image)["image"].unsqueeze(0)


def run_inference(model: Unet, image_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        try:
            output = model(image_tensor)
            print(output["obs"].shape)
            output = output["obs"].squeeze(1).squeeze(1)
            print(output.shape)
            k_predictions = output.numpy()
        except HTTPException as e:
            print(status_code=500, detail=f"Model prediction failed: {str(e)}")
    return k_predictions


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    global model

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(file_path)
    input_tensor = preprocess_image(image)
    print(input_tensor.mean())

    predictions = run_inference(model, input_tensor)
    mean_prediction = predictions.mean(axis=0).squeeze()

    output_dir = Path("static/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_image_path = output_dir / "input_image.png"
    cv2.imwrite(input_image_path, image)

    mean_prediction_path = output_dir / "mean_prediction.png"
    plot_image = input_tensor.squeeze(0).numpy().transpose(1, 2, 0) + 0.5
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(plot_image)
    ax.imshow(mean_prediction, alpha=0.5)

    plt.axis("off")
    fig.savefig(mean_prediction_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    individual_prediction_paths = []
    for i, prediction in enumerate(predictions):
        pred_path = output_dir / f"prediction_{i}.png"
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(plot_image)
        ax.imshow(prediction.squeeze(), alpha=0.5)
        plt.axis("off")
        fig.savefig(pred_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        individual_prediction_paths.append(
            f"/static/outputs/prediction_{i}.png"
        )

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "input_image": "/static/outputs/input_image.png",
            "mean_prediction": "/static/outputs/mean_prediction.png",
            "individual_predictions": individual_prediction_paths,
        },
    )

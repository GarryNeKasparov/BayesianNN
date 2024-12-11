import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pyro.infer import Predictive
from torchvision import transforms

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
        params = torch.load(MODEL_PATH)
        model.load_state_dict(torch.load("../models/raw.pt"))
        guide = params["guide"]
        model = Predictive(
            model, num_samples=10, guide=guide, return_sites=("obs", "_RETURN")
        )
        transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1]),
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
    return transform(image).unsqueeze(0)


def run_inference(model: Unet, image_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        try:
            output = model(image_tensor)
            print(output["obs"].size())
            output = output["obs"].squeeze(1).squeeze(1)
            k_predictions = output.numpy()
        except HTTPException as e:
            print(status_code=500, detail=f"Model prediction failed: {str(e)}")
    return k_predictions


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = Image.open(file_path).convert("RGB")
    input_tensor = preprocess_image(image)

    predictions = run_inference(model, input_tensor)
    mean_prediction = predictions.mean(axis=0).squeeze()

    output_dir = Path("static/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_image_path = output_dir / "input_image.png"
    image.save(input_image_path)

    mean_prediction_path = output_dir / "mean_prediction.png"
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(mean_prediction.numpy(), cmap="gray")
    ax.imshow(input_tensor.squeeze(0).numpy().transpose(1, 2, 0))
    plt.axis("off")
    fig.savefig(mean_prediction_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    individual_prediction_paths = []
    for i, prediction in enumerate(predictions):
        pred_path = output_dir / f"prediction_{i}.png"
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(prediction.squeeze().numpy(), cmap="gray")
        ax.imshow(input_tensor.squeeze(0).numpy().transpose(1, 2, 0))
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

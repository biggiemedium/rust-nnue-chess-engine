# handles GET requests from our rust engine

from fastapi import FastAPI
import logging

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chess NN Evaluator")


@app.on_event("startup")
async def startup_event():
    """Load the neural network model on server startup"""
    global evaluator

    try:
        logger.info("Loading neural network model...")

        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        logger.info(f"Model loaded on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
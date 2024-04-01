import io

from fastapi import APIRouter, Depends, File
from starlette.responses import Response
from ml.src.services.ml_service import AnomalyMLService
import pandas as pd


ml_router = APIRouter(
    tags=['ML'],
    prefix='/ml'
)

ml_service = AnomalyMLService()

@ml_router.post("/predict")
def get_segmentation_map(file: bytes = File(...)):
    csv_file = io.BytesIO(file)
    ml_service.preprocess(csv_file)
    final_data = ml_service.train()

    return Response(content=final_data.to_csv(), media_type="text/csv")




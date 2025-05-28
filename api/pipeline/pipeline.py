"""Pipeline."""

from typing import Callable, List
import logging
from pydantic import BaseModel
from ultralytics import YOLO
from .frame_sampler import StaticFrameSampler
from .object_detection import YOLOObjectDetection
from ..models import Prediction
from ..db import Database

logger = logging.getLogger()

class Pipeline(BaseModel):
    """Pipeline."""
    steps: List[Callable[[Prediction, Database], Prediction]]
    
    def __call__(self, prediction: Prediction, database: Database) -> Prediction:
        """Prediction."""
        logger.debug("Starting inference for prediction %s", prediction.id_)
        pred = prediction

        for step in self.steps:
            logger.debug(f"Running step %s", step.__class__.__name__)
            pred = step(pred, database)

        return pred

def get_pipeline() -> Pipeline:
    """Get pipeline."""
    return Pipeline(
        steps=[
            StaticFrameSampler(fps=15.0),
            YOLOObjectDetection(
                model=YOLO(
                    model="yolo11n.pt"
                )
            )
        ]
        )

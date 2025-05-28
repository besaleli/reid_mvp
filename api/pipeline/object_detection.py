"""Object Detection."""
from typing import List
import uuid
import logging
from pydantic import BaseModel, ConfigDict
from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
from ..models import Prediction, ObjectDetection
from ..db import Database

logger = logging.getLogger()

class YOLOObjectDetection(BaseModel):
    """YOLO Object Detection."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: YOLO
    batch_size: int = 32
    
    def __call__(self, prediction: Prediction, db: Database) -> Prediction:
        """Run."""
        if prediction.frames is None:
            raise ValueError("Frames must be present to do object detection.")

        prediction.detected_entities = []
        
        # convert inputs to RGB
        logger.debug("Converting frames to RGB...")
        imgs_rgb = [
            img.image.to_ndarray(format="rgb24") for img in prediction.frames
            ]
        
        # convert RGB to BGR, which YOLO expects
        logger.debug("Converting frames to BGR...")
        imgs_bgr = [
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs_rgb
        ]

        results: List[Results] = []
        
        for i in range(0, len(imgs_bgr), self.batch_size):
            batch = imgs_bgr[i:i+self.batch_size]
            results.extend(self.model(batch))
            logger.debug("Processed %s/%s frames", i, len(imgs_bgr))
        
        for i, result in enumerate(results):
            frame_id = prediction.frames[i].id_
            for r in result:
                if r.boxes is not None:
                    xyxy = r.boxes.xyxy
                    for j, box in enumerate(r.boxes):
                        if int(box.cls[0]) == 0:
                            obj = ObjectDetection(
                                id_=uuid.uuid4(),
                                frame_id=frame_id,
                                bbox=xyxy[j].tolist(),
                                confidence=float(box.conf[0]),
                            )

                            logger.debug("Adding detected object %s", obj.id_)
                            prediction.detected_entities.append(obj)

        return prediction

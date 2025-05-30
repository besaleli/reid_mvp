"""ReID Pipeline"""
from typing import List
import uuid
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
import pandas as pd
import time
import duckdb
import ruptures as rpt
import cv2
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
from torchvision import transforms
from torch.nn.functional import normalize
import torchreid
import clip

YOLO_PARAMETERS = {
    "classes": [0],
    "iou": 0.4,
    "conf": 0.5
}

conn = duckdb.connect(database="data/data.db")

class Frame(BaseModel):
    """Frame model"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    frame: Results
    timestamp_ms: int
    n_objects_detected: int

class Detection(BaseModel):
    """Frame Metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    detection_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    frame_id: uuid.UUID
    crop_bgr: np.ndarray
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    
class ObjectDetection(BaseModel):
    """Object Detection"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    conn: duckdb.DuckDBPyConnection

    yolo_model: str = "yolo11n.pt"
    yolo_parameters: dict

    osnet_model: str = "osnet_x1_0"
    osnet_batch_size: int = 32
    
    clip_model: str = "ViT-B/16"
    clip_batch_size: int = 32
    
    video_path: str
    video_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    video_uploaded_at: float = Field(default_factory=time.time)
    
    frames: List[Frame] = Field(default_factory=list)
    is_irregular: List[bool] = Field(default_factory=list)

    detections: List[Detection] = Field(default_factory=list)
    osnet_embeddings: List[np.ndarray] = Field(default_factory=list)
    clip_embeddings: List[np.ndarray] = Field(default_factory=list)
    
    @property
    def device(self) -> torch.device:
        return torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
    
    def run_yolo(self) -> 'ObjectDetection':
        """Run YOLO."""
        # load video
        cap = cv2.VideoCapture(self.video_path) 
    
        # this model would be deployed in a containerized service!
        model = YOLO(self.yolo_model)

        # this would probably be batched irl if doing offline inference
        # if online, then streamed
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get the current position of the video file in milliseconds
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # Inference
            result: Results = model(frame, **YOLO_PARAMETERS)[0]
            
            if len(result.boxes.xyxy) > 0:
                n_objects_detected = len(result.boxes.xyxy)
            else:
                n_objects_detected = 0

            # Append to states
            self.frames.append(
                Frame(
                    frame=result,
                    timestamp_ms=timestamp_ms,
                    n_objects_detected=n_objects_detected
                )
            )
        
        # deleting for memory reasons because I don't trust Python's garbo collector!
        del model
        del cap

        return self
    
    def extract_detections(self) -> 'ObjectDetection':
        for i, frame in enumerate(self.frames):
            result: Results = frame.frame
            confidences = result.boxes.conf.cpu().tolist()
            for conf, box in zip(confidences, result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                crop_bgr = result.orig_img[y1:y2, x1:x2]
                if crop_bgr.size == 0:
                    continue
                
                self.detections.append(
                    Detection(
                        frame_id=frame.frame_id,
                        crop_bgr=crop_bgr,
                        conf=conf,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2
                    )
                )

        return self
    
    def detect_irregular_frames(self) -> 'ObjectDetection':
        signal = [i.n_objects_detected for i in self.frames]
        
        algo = rpt.Pelt(model="l2").fit(np.array(signal))
        breakpoints = algo.predict(pen=6)
        
        for i in range(len(breakpoints)):
            start = 0 if i == 0 else breakpoints[i-1]
            end = breakpoints[i]
            frames = self.frames[start:end]
            n_objs_detected = [i.n_objects_detected for i in frames]
            mode = pd.Series(n_objs_detected).mode().to_list()[0]
            self.is_irregular += [i != mode for i in n_objs_detected]

        assert len(self.is_irregular) == len(self.frames)
        return self
    
    def compute_osnet_embeddings(self) -> 'ObjectDetection':
        """Compute OSNet embeddings"""
        device = self.device

        osnet_model = torchreid.models.build_model(
            name=self.osnet_model,
            num_classes=1000,
            loss="softmax",
            pretrained=True
        ).to(device).eval()
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        crop_rgbs = map(lambda i: cv2.cvtColor(i.crop_bgr, cv2.COLOR_BGR2RGB), self.detections)
        transformed_inputs = [transform(c) for c in crop_rgbs]
        
        embs = []
        
        for i in list(range(0, len(transformed_inputs), self.osnet_batch_size)):
            batch = torch.stack(transformed_inputs[i:i+self.osnet_batch_size]).to(device)
            with torch.no_grad():
                embeddings: torch.Tensor = osnet_model(batch)

            embs += embeddings
        
        embs = [i for i in normalize(torch.stack(embs), dim=1).cpu().numpy()]
        self.osnet_embeddings += embs
        
        del osnet_model
        del crop_rgbs
        del transformed_inputs
        
        assert len(self.detections) == len(self.osnet_embeddings)
        
        return self
    
    def compute_clip_embeddings(self) -> 'ObjectDetection':
        """Compute CLIP embeddings."""
        device = self.device
        model, preprocess = clip.load(self.clip_model)
        model = model.to(device).eval()

        inputs = [
            preprocess(
                Image.fromarray(
                    cv2.cvtColor(i.crop_bgr, cv2.COLOR_BGR2RGB),
                    mode="RGB"
                    )
                ) for i in self.detections
            ]
        
        embs = []
        for i in range(0, len(inputs), self.clip_batch_size):
            print(f"{i}/{len(inputs)}")
            batch = torch.stack(inputs[i:i+self.clip_batch_size]).to(device)
            with torch.no_grad():
                embs += model.encode_image(batch)
        
        embs = [i for i in normalize(torch.stack(embs), dim=1).cpu().numpy()]
        self.clip_embeddings += embs
        
        assert len(self.clip_embeddings) == len(self.detections)
        
        del model
        del preprocess
        del inputs
        del embs
        
        return self
    
    def create_video_dataframe(self) -> pd.DataFrame:
        data = [
            {
                "id": self.video_id,
                "filepath": self.video_path,
                "duration_ms": max(self.frames, key=lambda i: i.timestamp_ms).timestamp_ms,
                "n_frames": len(self.frames),
                "uploaded_at": self.video_uploaded_at
            }
        ]
        
        return pd.DataFrame(data)
    
    def create_frame_dataframe(self) -> pd.DataFrame:
        data = [
            {
                "id": f.frame_id,
                "video_id": self.video_id,
                "video_frame_index": frame_index,
                "n_objects_detected": f.n_objects_detected,
                "is_irregular": self.is_irregular[frame_index],
                "timestamp_ms": f.timestamp_ms,
            } for frame_index, f in enumerate(self.frames)
        ]
        
        return pd.DataFrame(data)
    
    def create_detection_dataframe(self) -> pd.DataFrame:
        data = [
            {
                "id": d.detection_id,
                "frame_id": d.frame_id,
                "conf": d.conf,
                "x1": d.x1,
                "y1": d.y1,
                "x2": d.x2,
                "y2": d.y2,
                "osnet_embedding": self.osnet_embeddings[i].tolist(),
                "clip_embedding": self.clip_embeddings[i].tolist(),
            } for i, d in enumerate(self.detections)
        ]
        
        return pd.DataFrame(data)

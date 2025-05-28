"""Data models."""

from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, ConfigDict
import av

class HasID(BaseModel):
    """ID."""
    id_: UUID


class Video(HasID):
    """Video model."""
    path: str


class Frame(HasID):
    """Frame."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: float
    
    # NOTE: this would not be a field in the data model. Assuming we can keep the pipeline
    # idempotent, we ideally shouldn't store this.
    image: av.VideoFrame


class ObjectDetection(HasID):
    """Object Detection."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_id: UUID
    bbox: List[float]
    confidence: float


class Prediction(HasID):
    """State dictionary used for a pipeline run."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_id: UUID
    created_at: datetime
    frames: Optional[List[Frame]] = None
    detected_entities: Optional[List[ObjectDetection]] = None
    
    @classmethod
    def new(cls, video_id: UUID) -> 'Prediction':
        """Create new prediction."""
        return cls(id_=uuid4(), video_id=video_id, created_at=datetime.now())

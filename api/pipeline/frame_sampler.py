"""Frame Sampler."""

import logging
import uuid
import av
from pydantic import BaseModel
from ..models import Prediction, Frame
from ..db import Database

logger = logging.getLogger()

class StaticFrameSampler(BaseModel):
    """Static frame sampler."""
    fps: float = 15.0

    def __call__(self, prediction: Prediction, db: Database) -> Prediction:
        """Run."""

        # set prediction frames to empty list
        prediction.frames = []

        container = av.open(db.get_video(prediction.video_id).path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        target_interval = 1.0 / self.fps # seconds
        last_ts = -target_interval # ensure first frame is sampled

        count = 0

        for frame in container.decode(video=0):
            # get timestamp
            ts = float(frame.pts * stream.time_base)
            
            # if we've passed the target interval, capture a new frame
            if ts - last_ts >= target_interval:
                prediction.frames.append(
                    Frame(
                        id_=uuid.uuid4(),
                        timestamp=ts,
                        image=frame
                        )
                    )
                last_ts = ts
                count += 1

        logger.debug("%s frames captured for video id %s", count, prediction.video_id)

        return prediction

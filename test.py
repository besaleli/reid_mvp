"""Test on videos."""

import uuid
import logging
from api.pipeline.pipeline import get_pipeline
from api.models import Prediction, Video
from api.db import Database

logging.basicConfig(level=logging.DEBUG)

video_1 = Video(id_=uuid.uuid4(), path="data/video_1.mp4")

db = Database.new()
db.add_video(video_1)

prediction = Prediction.new(video_1.id_)

pipe = get_pipeline()

pred = pipe(prediction, db)

import os
import shutil
import logging
import duckdb
import threading
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from api.anomaly_detection import AppearanceAnomalyDetector
from api.intervideo_reid import CrossVideoReID
from api.intravideo_reid import IntraVideoObjectIdPipeline
from api.obj_detection import ObjectDetection
from api.scene_generation import SceneSegmentationPipeline

# ========== Globals ==========
UPLOAD_DIR = "uploads"
DB_PATH = "data/final.db"
YOLO_PARAMETERS = {"classes": [0], "iou": 0.4, "conf": 0.5}
os.makedirs(UPLOAD_DIR, exist_ok=True)

_db_lock = threading.Lock()
_db = duckdb.connect(DB_PATH)

def get_db():
    return _db

def with_db_lock(func):
    def wrapper(*args, **kwargs):
        with _db_lock:
            return func(*args, **kwargs)
    return wrapper

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(video_id)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class VideoAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return msg, {"extra": {"video_id": self.extra.get("video_id", "-")}}

def get_logger(video_id=None):
    return VideoAdapter(logging.getLogger(__name__), {"video_id": video_id})

# ========== ThreadPoolExecutor ==========
executor = ThreadPoolExecutor(max_workers=1)

def enqueue_video(video_path, video_id):
    executor.submit(run_pipeline, video_path, video_id)

# ========== Pipeline ==========
@with_db_lock
def run_pipeline(video_path: str, video_id: str):
    logger = get_logger(video_id)
    db = get_db()
    logger.info("Starting video processing pipeline")

    try:
        with open("schema.sql") as f:
            db.sql(f.read())
            db.commit()
        logger.info("Schema initialized")
    except Exception as e:
        logger.error(f"Schema init failed: {e}")
        return

    try:
        preds = (
            ObjectDetection(
                conn=db,
                yolo_parameters=YOLO_PARAMETERS,
                video_path=video_path,
            )
            .run_yolo()
            .extract_detections()
            .detect_irregular_frames()
            .compute_osnet_embeddings()
            .compute_clip_embeddings()
        )
        logger.info("Object detection completed")
    except Exception as e:
        logger.error(f"Object detection failed: {e}")
        return

    try:
        video_df = preds.create_video_dataframe()
        frame_df = preds.create_frame_dataframe()
        detection_df = preds.create_detection_dataframe()

        db.register("video_df_temp", video_df)
        db.register("frame_df_temp", frame_df)
        db.register("detection_df_temp", detection_df)

        db.execute("insert into video select * from video_df_temp")
        db.execute("insert into frame select * from frame_df_temp")
        db.execute("insert into detection select * from detection_df_temp")

        db.unregister("video_df_temp")
        db.unregister("frame_df_temp")
        db.unregister("detection_df_temp")
        db.commit()
        logger.info("Detection data inserted")
    except Exception as e:
        logger.error(f"Insertion failed: {e}")
        return

    try:
        pipe = IntraVideoObjectIdPipeline(conn=db, video_id=video_id).create_clusters()
        db.register("intravideo_object_ids_temp", pipe.df)
        db.execute("insert into intravideo_object_ids select * from intravideo_object_ids_temp")
        db.unregister("intravideo_object_ids_temp")
        db.commit()
        logger.info("Intra-video ReID complete")

        pipe = SceneSegmentationPipeline(conn=db, video_id=video_id).segment()
        db.register("scene_temp", pipe.df)
        db.execute("insert into scene select * from scene_temp")
        db.unregister("scene_temp")
        db.commit()
        logger.info("Scene segmentation complete")

        results = CrossVideoReID(conn=db, threshold=0.9).compute_crossvideo_reids().df
        db.register("intervideo_object_ids_temp", results)
        db.execute("insert into intervideo_object_ids select * from intervideo_object_ids_temp")
        db.unregister("intervideo_object_ids_temp")
        db.commit()
        logger.info("Inter-video ReID complete")
    except Exception as e:
        logger.error(f"ReID/Scene failed: {e}")
        return

    logger.info("Video processing complete")

# ========== API ==========
app = FastAPI(title="Mini Re-ID Service")

@app.post("/videos")
async def upload_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail="Unsupported video format")

    video_id = str(uuid4())
    dest = os.path.join(UPLOAD_DIR, f"{video_id}_{file.filename}")

    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger = get_logger(video_id)
    logger.info(f"Video uploaded: {file.filename}")

    enqueue_video(dest, video_id)
    return {"video_id": video_id, "status": "queued"}

@app.get("/persons")
@with_db_lock
def get_persons():
    conn = get_db()
    person_ids = [row[0] for row in conn.execute("SELECT DISTINCT person_id FROM intervideo_object_ids").fetchall()]
    result = []

    for pid in person_ids:
        segments = conn.execute(f"""
            SELECT video_id, MIN(start_ts) as start_ts, MAX(end_ts) as end_ts
            FROM detection
            WHERE object_id = '{pid}'
            GROUP BY video_id
        """).fetchall()

        appearances = [{"video_id": v, "start_ts": float(s), "end_ts": float(e)} for v, s, e in segments]
        result.append({"person_id": pid, "appearances": appearances})

    return JSONResponse(content=result)

@app.get("/persons/{person_id}/anomalies")
@with_db_lock
def get_anomalies(person_id: str):
    conn = get_db()
    video_ids = conn.execute(f"""
        SELECT DISTINCT video_id
        FROM detection
        WHERE object_id = '{person_id}'
    """).fetchall()

    result = []
    logger = get_logger(person_id)
    logger.info("Running real-time anomaly detection")

    for (video_id,) in video_ids:
        try:
            detector = AppearanceAnomalyDetector(conn=conn, video_id=video_id).detect()
            df = detector.df
            df = df[df["object_id"] == person_id]

            for _, row in df.iterrows():
                result.append({
                    "video_id": row["video_id"],
                    "timestamp": float(row["timestamp"]),
                    "type": row["type"]
                })
        except Exception as e:
            logger.error(f"Anomaly detection failed for {video_id}: {e}")

    return JSONResponse(content=result)

import uuid
import time
import numpy as np
import pandas as pd
import duckdb
import ruptures as rpt
from scipy.ndimage import binary_opening, binary_closing
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

SCENE_QUERY = """
select
    v.id as video_id,
    f.video_frame_index,
    f.timestamp_ms,
    d.id as detection_id,
    r.cluster_id
from intravideo_object_ids r
left join detection d on d.id = r.detection_id
left join frame f on f.id = d.frame_id
left join video v on v.id = f.video_id
where
    v.id = ?
    and not r.is_bad_frame
"""

class SceneSegmentationPipeline(BaseModel):
    """Scene segmentation per cluster_id via ruptures change point detection."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    conn: duckdb.DuckDBPyConnection
    video_id: uuid.UUID
    created_at: float = Field(default_factory=time.time)
    df: Optional[pd.DataFrame] = None

    def segment(self) -> "SceneSegmentationPipeline":
        df = self.conn.execute(SCENE_QUERY, [self.video_id.hex]).df()
        self.df = df
        if df.empty:
            return self

        max_frame = df['video_frame_index'].max()
        n_frames = max_frame + 1
        cluster_ids = df['cluster_id'].unique()

        records = []

        for cluster_id in cluster_ids:
            signal = np.zeros(n_frames, dtype=np.uint8)
            indices = df[df.cluster_id == cluster_id]["video_frame_index"].values
            signal[indices] = 1

            smoothed = binary_closing(signal, structure=np.ones(10))
            smoothed = binary_opening(smoothed, structure=np.ones(10))

            algo = rpt.Pelt(model="l2", min_size=100).fit(smoothed)
            breakpoints = algo.predict(pen=20)

            min_length = 150
            filtered_cps = [cp for i, cp in enumerate(breakpoints)
                            if i == 0 or (cp - breakpoints[i - 1]) > min_length]

            prev = 0
            for bucket_index, curr in enumerate(filtered_cps):
                start_frame = prev
                end_frame = curr
                prev = curr

                frame_range = df[
                    (df.video_frame_index >= start_frame) &
                    (df.video_frame_index < end_frame) &
                    (df.cluster_id == cluster_id)
                ]

                if frame_range.empty:
                    continue

                start_ts = frame_range['timestamp_ms'].min()
                end_ts = frame_range['timestamp_ms'].max()

                records.append({
                    "video_id": self.video_id,
                    "cluster_id": cluster_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "bucket_index": bucket_index,
                    "start_timestamp_ms": float(start_ts),
                    "end_timestamp_ms": float(end_ts),
                    "n_frames": end_frame - start_frame
                })

        self.df = pd.DataFrame(records)
        return self

db = duckdb.connect("data/data.db")

video_ids = [i[0] for i in db.sql("select distinct id from video").fetchall()]

for video_id in video_ids:
    pipe = SceneSegmentationPipeline(conn=db, video_id=video_id).segment()
    db.register("scene_temp", pipe.df)
    db.execute("insert into scene select * from scene_temp")
    db.unregister("scene_temp")
    db.commit()

db.close()

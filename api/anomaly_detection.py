"""Anomaly detection."""
from typing import Optional
import duckdb
import uuid
import pandas as pd
import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

APPEARANCE_QUERY = """
-- Combine all CTEs under a single WITH clause
with
final as (
    select
        s.video_id,
        v.uploaded_at,
        f.video_frame_index,
        s.bucket_index,
        f.timestamp_ms,
        s.cluster_id,
        d.clip_embedding
    from scene s
    left join frame f on f.video_id = s.video_id
    left join detection d on d.frame_id = f.id
    left join intravideo_object_ids i on i.detection_id = d.id
    left join video v on s.video_id = v.id
    where
        s.video_id = ?
        and f.video_frame_index between s.start_frame and s.end_frame
        and not i.is_bad_frame
        and s.cluster_id = i.cluster_id
    qualify count(*) over (partition by s.video_id, s.cluster_id, s.bucket_index) > 10
),
prev_id as (
    SELECT id
    FROM video
    WHERE uploaded_at < (
        SELECT uploaded_at
        FROM video
        WHERE id = ?
    )
    ORDER BY uploaded_at DESC
    LIMIT 1
),
data as (
    select
        s.video_id,
        v.uploaded_at,
        f.video_frame_index,
        s.bucket_index,
        f.timestamp_ms,
        s.cluster_id,
        d.clip_embedding
    from scene s
    left join frame f on f.video_id = s.video_id
    left join detection d on d.frame_id = f.id
    left join intravideo_object_ids i on i.detection_id = d.id
    left join video v on s.video_id = v.id
    where
        s.video_id in (select * from prev_id)
        and f.video_frame_index between s.start_frame and s.end_frame
        and not i.is_bad_frame
        and s.cluster_id = i.cluster_id
    qualify count(*) over (partition by s.video_id, s.cluster_id, s.bucket_index) > 10
),
most_recent_frames as (
    select distinct bucket_index from data order by 1 desc limit ?
)

-- First query
select
    f.video_id,
    f.uploaded_at,
    f.video_frame_index,
    f.bucket_index,
    f.timestamp_ms,
    l.cluster_id,
    f.clip_embedding
from final f
left join latest_global_object_ids l on f.cluster_id = l.intravideo_cluster_id and f.video_id = l.video_id

union all

-- Second query
select
    f.video_id,
    f.uploaded_at,
    f.video_frame_index,
    f.bucket_index,
    f.timestamp_ms,
    l.cluster_id,
    f.clip_embedding
from data f
left join latest_global_object_ids l on f.cluster_id = l.intravideo_cluster_id and f.video_id = l.video_id
where
    f.bucket_index in (select * from most_recent_frames)
"""

class AppearanceAnomalyDetector(BaseModel):
    """Detect appearance anomalies across all clusters in a video."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    conn: duckdb.DuckDBPyConnection
    video_id: uuid.UUID
    threshold: float = 0.48
    stratified_sample_k: int = 100
    prev_scenes_considered: int = 2
    df: pd.DataFrame = pd.DataFrame()
    
    def _preprocess_df(self) -> Optional[pd.DataFrame]:
        """Add inter-arrival times"""
        df = self.conn.execute(APPEARANCE_QUERY, [self.video_id.hex, self.video_id.hex, self.prev_scenes_considered]).df()
        df['vb_index'] = list(zip(df['video_id'], df['bucket_index']))
        
        dfs = []

        for _, dff in df.groupby(["video_id", "cluster_id"]):
            c = dff.sort_values("video_frame_index")
            c["iat"] = c["timestamp_ms"].diff().fillna(1)
            c["iat_norm"] = c.groupby("bucket_index")["iat"].transform(lambda x: (x + 1e-3) / (x.sum() + 1e-6))
            c = c.groupby("bucket_index").filter(lambda g: g.shape[0] > 10)
            c = (
                c.sort_values("iat_norm", ascending=False)
                .groupby("bucket_index")
                .head(self.stratified_sample_k)
                .sort_values("video_frame_index")
            )
            
            if not c.empty:
                dfs.append(c)
        
        return pd.concat(dfs)
            

    def detect(self) -> "AppearanceAnomalyDetector":
        df = self._preprocess_df()

        results = [self._process_cluster(dff) for _, dff in df.groupby("cluster_id")]
        self.df = pd.concat([r for r in results if r is not None], ignore_index=True) if results else pd.DataFrame()
        return self

    def _process_cluster(self, df: pd.DataFrame) -> pd.DataFrame | None:
        pca_50 = PCA(n_components=50)
        df["clip_pca_50"] = list(pca_50.fit_transform(df["clip_embedding"].to_list()))

        appearance_buckets = []
        appearance_matrices = []
        distances = [0]

        for (vid, bidx), group in df.sort_values(by=["uploaded_at", "bucket_index"]).groupby("vb_index"):
            appearance_buckets.append((vid, bidx))
            appearance_matrices.append(np.stack(group["clip_pca_50"].to_list()))

        for i in range(1, len(appearance_buckets)):
            prev, curr = appearance_matrices[i - 1], appearance_matrices[i]
            d = self._energy_distance_cosine(prev, curr)
            distances.append(d)

        anomaly_map = {
            b: bool(d > self.threshold)
            for b, d in zip(appearance_buckets, distances)
        }
        
        distance_map = {
            b: d for b, d in zip(appearance_buckets, distances)
        }

        df['energy_distance'] = df['vb_index'].map(distance_map)
        df["is_anomaly"] = df["vb_index"].map(anomaly_map)
        df = df[df['video_id'] == self.video_id]

        return df

    def _energy_distance_cosine(self, X: np.ndarray, Y: np.ndarray) -> float:
        N, M = len(X), len(Y)
        cross = np.sum(cdist(X, Y, metric="cosine")) / (N * M)
        xx = np.sum(cdist(X, X, metric="cosine")) / (N * N)
        yy = np.sum(cdist(Y, Y, metric="cosine")) / (M * M)
        return 2 * cross - xx - yy

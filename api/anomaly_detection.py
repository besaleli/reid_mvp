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
with final as (
    select
        s.video_id,
        f.video_frame_index,
        s.bucket_index,
        f.timestamp_ms,
        s.cluster_id,
        d.clip_embedding
    from scene s
    left join frame f on f.video_id = s.video_id
    left join detection d on d.frame_id = f.id
    left join intravideo_object_ids i on i.detection_id = d.id
    where
        s.video_id = ?
        and f.video_frame_index between s.start_frame and s.end_frame
        and not i.is_bad_frame
        and s.cluster_id = i.cluster_id
    qualify count(*) over (partition by s.video_id, s.cluster_id, s.bucket_index) > 10
)

select
    f.video_id,
    f.video_frame_index,
    f.bucket_index,
    f.timestamp_ms,
    l.cluster_id, -- use global cluster id
    f.clip_embedding
from final f
left join latest_global_object_ids l on f.cluster_id=l.intravideo_cluster_id and f.video_id=l.video_id
"""

def normalize_buckets(prev: list, curr: list):
    """Correct arbitrary bucket identification and merge two from separate dfs if necessary"""
    prev = [(0, x) for x in prev]
    curr = [(1, x) for x in curr]
    
    l = sorted(prev + curr)
    
    vals = {}
    count = 0
    
    for x1, y1 in l:
        if (x1, y1) not in vals:
            vals[(x1, y1)] = count
            count += 1

    return [vals[i] for i in l]

class AppearanceAnomalyDetector(BaseModel):
    """Detect appearance anomalies across all clusters in a video."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    conn: duckdb.DuckDBPyConnection
    video_id: uuid.UUID
    threshold: float = 0.48
    stratified_sample_k: int = 100
    df: pd.DataFrame = pd.DataFrame()
    
    def _preprocess_df(self) -> Optional[pd.DataFrame]:
        """Add inter-arrival times"""
        df = self.conn.execute(APPEARANCE_QUERY, [self.video_id.hex]).df()
        
        if df is None:
            return None
        
        dfs = []
        
        for cluster, dff in df.groupby("cluster_id"):
            c = dff.sort_values("video_frame_index")
            c["iat"] = c["timestamp_ms"].diff().fillna(0)
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
        df["vb_index"] = [(x, y) for x, y in zip(df['video_id'], df['bucket_index'])]

        appearance_buckets = []
        appearance_matrices = []
        distances = [0]

        for (vid, bidx), group in df.groupby("vb_index"):
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
        return df

    def _energy_distance_cosine(self, X: np.ndarray, Y: np.ndarray) -> float:
        N, M = len(X), len(Y)
        cross = np.sum(cdist(X, Y, metric="cosine")) / (N * M)
        xx = np.sum(cdist(X, X, metric="cosine")) / (N * N)
        yy = np.sum(cdist(Y, Y, metric="cosine")) / (M * M)
        return 2 * cross - xx - yy

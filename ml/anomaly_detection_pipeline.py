import duckdb
import uuid
import pandas as pd
import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

APPEARANCE_QUERY = """
select
    s.video_id,
    s.bucket_index,
    s.start_frame,
    s.end_frame,
    f.video_frame_index,
    f.timestamp_ms,
    gid.cluster_id,
    d.clip_embedding
from scene s
left join frame f on f.video_id = s.video_id
left join detection d on d.frame_id = f.id
left join intravideo_object_ids i on i.detection_id = d.id
left join latest_global_object_ids gid on f.video_id=gid.video_id
where
    s.video_id = ?
    and s.cluster_id = ?
    and f.video_frame_index between s.start_frame and s.end_frame
    and not i.is_bad_frame
    and s.cluster_id = i.cluster_id
    and i.cluster_id=gid.intravideo_cluster_id
qualify count(*) over (partition by s.video_id, s.cluster_id, s.bucket_index) > 10
"""

class AppearanceAnomalyDetector(BaseModel):
    """Detect appearance anomalies across all clusters in a video."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    conn: duckdb.DuckDBPyConnection
    video_id: uuid.UUID
    threshold: float = 0.48
    stratified_sample_k: int = 100
    df: pd.DataFrame = pd.DataFrame()

    def detect(self) -> "AppearanceAnomalyDetector":
        cluster_ids = [
            row[0] for row in
            self.conn.execute("select distinct cluster_id from scene where video_id = ?", [self.video_id]).fetchall()
        ]

        results = [self._process_cluster(cid) for cid in cluster_ids]
        self.df = pd.concat([r for r in results if r is not None], ignore_index=True) if results else pd.DataFrame()
        return self

    def _process_cluster(self, cluster_id: int) -> pd.DataFrame | None:
        df = self.conn.execute(APPEARANCE_QUERY, [self.video_id.hex, cluster_id]).df()

        if df.empty:
            return None

        df = df.sort_values("video_frame_index")
        df["iat"] = df["timestamp_ms"].diff().fillna(0)
        df["iat_norm"] = df.groupby("bucket_index")["iat"].transform(lambda x: (x + 1e-3) / (x.sum() + 1e-6))
        df = df.groupby("bucket_index").filter(lambda g: g.shape[0] > 10)
        df = (
            df.sort_values("iat_norm", ascending=False)
            .groupby("bucket_index")
            .head(self.stratified_sample_k)
            .sort_values("video_frame_index")
        )

        if df.empty:
            return None

        pca_50 = PCA(n_components=50)
        df["clip_pca_50"] = list(pca_50.fit_transform(df["clip_embedding"].to_list()))

        appearance_buckets = []
        appearance_matrices = []
        distances = [0]

        for bidx, group in df.groupby("bucket_index"):
            appearance_buckets.append(bidx)
            appearance_matrices.append(np.stack(group["clip_pca_50"].to_list()))

        for i in range(1, len(appearance_buckets)):
            prev, curr = appearance_matrices[i - 1], appearance_matrices[i]
            d = self._energy_distance_cosine(prev, curr)
            distances.append(d)

        anomaly_map = {
            b: bool(d > self.threshold)
            for b, d in zip(appearance_buckets, distances)
        }

        df["is_anomaly"] = df["bucket_index"].map(anomaly_map)
        return df

    def _energy_distance_cosine(self, X: np.ndarray, Y: np.ndarray) -> float:
        N, M = len(X), len(Y)
        cross = np.sum(cdist(X, Y, metric="cosine")) / (N * M)
        xx = np.sum(cdist(X, X, metric="cosine")) / (N * N)
        yy = np.sum(cdist(Y, Y, metric="cosine")) / (M * M)
        return 2 * cross - xx - yy

db = duckdb.connect("data/data.db")
video_ids = [i[0] for i in db.sql("select distinct id from video").fetchall()]

for video_id in video_ids:
    detector = AppearanceAnomalyDetector(conn=db, video_id=video_id).detect()
    df = detector.df
    print(df.columns)
    print(df[df['is_anomaly']]['bucket_index'].value_counts())
    for i, dff in df.groupby("cluster_id"):
        print(dff.groupby('bucket_index')['is_anomaly'].value_counts())

db.close()

"""Clustering pipeline"""
from typing import Optional
import uuid
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pydantic import BaseModel, ConfigDict
import duckdb

GET_DATA = """
with data as (
    select
        d.frame_id,
        d.id,
        d.osnet_embedding,
        f.n_objects_detected,
        f.is_irregular,
        d.conf
    from detection d
    left join frame f on d.frame_id=f.id
    where f.video_id = '{}'
)

select
    *
from data
"""

CLEAN_CLUSTERS = """
with cpf as (
    select
        frame_id,
        count(distinct cluster_id_raw) as n_distinct_clusters_per_frame
    from clusters
    group by 1
    order by 1
),

bad_frames as (
    select
        clusters.frame_id
    from clusters
    left join cpf on clusters.frame_id = cpf.frame_id
    where
        n_objects_detected > 1
        and n_objects_detected > n_distinct_clusters_per_frame
    qualify count(*) over (partition by clusters.frame_id) > 1
),

cleaned as (
    select
        *,
        frame_id in (select * from bad_frames) as is_bad_frame,
        case
            when frame_id in (select * from bad_frames) then -1
            else cluster_id_raw
        end as cluster_id
    from clusters
)

select
    uuidv4() as id,
    cleaned.id as person_detection_id,
    cleaned.cluster_id,
    cleaned.is_bad_frame
from cleaned
left join detection on cleaned.id=detection.id
"""

class ClusteringPipeline(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    conn: duckdb.DuckDBPyConnection
    video_id: uuid.UUID
    df: Optional[pd.DataFrame] = None
    
    def create_clusters(self) -> 'ClusteringPipeline':
        """Create clusters."""
        query = GET_DATA.format(self.video_id.hex)
        df = self.conn.sql(query).df()
        df['conf_under_p5'] = df['conf'] < np.quantile(df['conf'].to_numpy(), 0.05)
        df = df[df.apply(lambda row: not (row['conf_under_p5'] or row['is_irregular']), axis=1)]
        pca_50 = PCA(n_components=50).fit_transform(df['osnet_embedding'].to_list())
        df['cluster_id_raw'] = KMeans(n_clusters=2).fit_predict(pca_50)
        self.conn.register("clusters", df)
        self.df = self.conn.sql(CLEAN_CLUSTERS).df()
        self.conn.unregister("clusters")
        
        return self
        
db = duckdb.connect("data/data.db")

video_ids = [i[0] for i in db.sql("select distinct id from video").fetchall()]

for video_id in video_ids:
    pipe = ClusteringPipeline(conn=db, video_id=video_id).create_clusters()
    db.register("reid_cluster_temp", pipe.df)
    db.execute("insert into reid_cluster select * from reid_cluster_temp")
    db.unregister("reid_cluster_temp")
    db.commit()

db.close()

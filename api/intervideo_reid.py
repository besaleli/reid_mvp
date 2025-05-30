"""Intervideo ReID pipeline"""
from typing import Optional
import uuid
import time
from pydantic import BaseModel, ConfigDict, Field
import duckdb
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering

QUERY = """
-- doing a stratified sample across video_id and cluster_id
-- databases aren't really great for complex stratified sampling like this
-- so we're just going to do every 10th record. we can think about
-- more complex sampling techniques and efficient implementations once
-- we identify exactly what the limitations of this are (there really may not be any)
with numbered as (
    select
        v.id as video_id,
        v.filepath,
        v.uploaded_at,
        d.osnet_embedding,
        ioi.cluster_id as intravideo_cluster_id,
        row_number() over (
            partition by v.id, ioi.cluster_id
            order by f.id -- or d.id, whatever is consistent
        ) as rn
    from video v
    left join frame f on v.id = f.video_id
    left join detection d on f.id = d.frame_id
    left join intravideo_object_ids ioi on d.id = ioi.detection_id
    where
        not (f.is_irregular or ioi.is_bad_frame)
)
select
    video_id,
    regexp_extract(filepath, '(video_[0-9])', 1) as video_name,
    uploaded_at,
    osnet_embedding,
    intravideo_cluster_id
from numbered
where rn % 5 = 1 -- every 5th record; adjust as needed
"""

class CrossVideoReID(BaseModel):
    """Cross-video ReID."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    conn: duckdb.DuckDBPyConnection
    threshold: float = 0.9
    created_at: float = Field(default_factory=time.time)
    df: Optional[pd.DataFrame] = None
    
    def compute_crossvideo_reids(self) -> 'CrossVideoReID':
        """Compute cross-video reIDs"""
        df = self.conn.query(QUERY).df()
        
        matches = []
        
        # if only one video in the database, no need to run clustering
        if df['video_id'].nunique() <= 1:
            for (vid, pid), group in df.groupby(['video_id', 'intravideo_cluster_id']):
                matches.append({
                    'id': uuid.uuid4(),
                    'video_id': vid,
                    'intravideo_cluster_id': pid,
                    'cluster_id': pid,  # just map to itself
                    'confidence_score': 1.0,
                    'is_match': True,
                    'threshold': self.threshold,
                    'created_at': self.created_at
                })

        else:
            # compute pca50
            pca_50 = PCA(n_components=50).fit_transform(df['osnet_embedding'].to_list())

            # run clustering
            df['pred_cluster'] = SpectralClustering(n_clusters=2).fit_predict(pca_50)

            gt_groups = df.groupby(['video_id', 'intravideo_cluster_id'])

            for (vid, pid), group in gt_groups:
                gt_indices = set(group.index)
                best_score = 0
                best_cluster = None

                for cluster_id in df['pred_cluster'].unique():
                    pred_indices = set(df[df['pred_cluster'] == cluster_id].index)
                    intersection = gt_indices & pred_indices
                    inclusion_score = len(intersection) / len(gt_indices)

                    if inclusion_score > best_score:
                        best_score = inclusion_score
                        best_cluster = cluster_id

                matches.append({
                    'id': uuid.uuid4(),
                    'video_id': vid,
                    'intravideo_cluster_id': pid,
                    'cluster_id': best_cluster,
                    'confidence_score': best_score,
                    'is_match': best_score > self.threshold,
                    'threshold': self.threshold,
                    'created_at': self.created_at
                })
            
        self.df = pd.DataFrame(matches)
        
        return self

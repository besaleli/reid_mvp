"""
Database.

NOTE: This should be replaced by an actual relational database, possibly with vector search capabilities.
I've abstracted what I think would be the SQL queries into functions. If I have time, I'll add a sample
SQL schema and queries that correlate to each function.

Personally, I would use pgvector for this. If making a Python API, I'd use a SQLAlchemy + Alembic stack. If making a Rust API, I'd use diesel-rs.

See data models in `api/db.py`
"""

from typing import List
from uuid import UUID
from pydantic import BaseModel, Field
from .models import Video

class Database(BaseModel):
    """Database."""
    videos: List[Video]
    
    @classmethod
    def new(cls) -> 'Database':
        return cls(
            videos=[]
        )

    def add_video(self, video: Video):
        self.videos.append(video)

    def get_video(self, video_id: UUID):
        candidates = [vid for vid in self.videos if video_id == vid.id_]
        
        if len(candidates) < 1:
            raise ValueError(f"Video ID {video_id} not found.")
        
        if len(candidates) > 1:
            raise ValueError(f"Duplicate videos detected. This should not happen.")

        return candidates[0]

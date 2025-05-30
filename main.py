import duckdb
from api.anomaly_detection import AppearanceAnomalyDetector
from api.intervideo_reid import CrossVideoReID
from api.intravideo_reid import IntraVideoObjectIdPipeline
from api.obj_detection import ObjectDetection
from api.scene_generation import SceneSegmentationPipeline

YOLO_PARAMETERS = {
    "classes": [0],
    "iou": 0.4,
    "conf": 0.5
}

videos = ["data/video_1.mp4"]

def main():
    # run object detection
    db = duckdb.connect("data/anomaly_detection.db")
    
    with open('schema.sql') as f:
        ctq = f.read()

    db.sql(ctq)
    db.commit()
    
    print("running object detection")
    for vpath in videos:
        preds = (
            ObjectDetection(
                conn=db,
                yolo_parameters=YOLO_PARAMETERS,
                video_path=vpath,
            )
            .run_yolo()
            .extract_detections()
            .detect_irregular_frames()
            .compute_osnet_embeddings()
            .compute_clip_embeddings()
        )

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
        
        print("Running intravideo object reid")
        video_ids = [i[0] for i in db.sql("select distinct id from video").fetchall()]

        for video_id in video_ids:
            pipe = IntraVideoObjectIdPipeline(conn=db, video_id=video_id).create_clusters()
            db.register("intravideo_object_ids_temp", pipe.df)
            db.execute("insert into intravideo_object_ids select * from intravideo_object_ids_temp")
            db.unregister("intravideo_object_ids_temp")
            db.commit()
        
        print('running scene segmentation')
        for video_id in video_ids:
            pipe = SceneSegmentationPipeline(conn=db, video_id=video_id).segment()
            db.register("scene_temp", pipe.df)
            db.execute("insert into scene select * from scene_temp")
            db.unregister("scene_temp")
            db.commit()
        
        print('running intervideo reid')
        results = (
            CrossVideoReID(conn=db, threshold=0.9)
            .compute_crossvideo_reids()
            .df
            )

        db.register("intervideo_object_ids_temp", results)
        db.execute("insert into intervideo_object_ids select * from intervideo_object_ids_temp")
        db.unregister("intervideo_object_ids_temp")
        
        print('running anomaly detection')
        for video_id in video_ids:
            detector = AppearanceAnomalyDetector(conn=db, video_id=video_id).detect()
            df = detector.df
            print(df.columns)
            print(df[df['is_anomaly']]['bucket_index'].value_counts())
        
        db.close()


if __name__ == "__main__":
    main()

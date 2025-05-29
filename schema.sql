create table if not exists video (
    id uuid not null,
    name varchar not null,
    duration_ms float not null,
    n_frames int not null,
    results_path varchar not null
);

create table if not exists frame (
    id uuid not null,
    video_id uuid not null,
    video_frame_index int not null,
    n_objects_detected int not null,
    is_irregular_detection bool not null,
    timestamp_ms float not null
);

create table if not exists person_detection (
    id uuid not null,
    frame_id uuid not null,
    conf float not null,
    x1 float not null,
    y1 float not null,
    x2 float not null,
    y2 float not null,
    osnet_embedding float[512] not null,
    clip_embedding float[512] not null
);

create table if not exists reid_cluster (
    id uuid not null,
    person_detection_id uuid not null,
    cluster_id int not null,
    is_bad_frame bool not null
);

create table if not exists video (
    id uuid not null,
    filepath varchar not null,
    duration_ms int not null,
    n_frames int not null,
    uploaded_at float not null
);

create table if not exists frame (
    id uuid not null,
    video_id uuid not null,
    video_frame_index int not null,
    n_objects_detected int not null,
    is_irregular bool not null,
    timestamp_ms float not null
);

create table if not exists detection (
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

create table if not exists intravideo_object_ids (
    id uuid not null,
    detection_id uuid not null,
    cluster_id int not null,
    is_bad_frame bool not null
);

create table if not exists intervideo_object_ids (
    id uuid not null,
    video_id uuid not null,
    intravideo_cluster_id int not null,
    cluster_id int not null,
    confidence_score float not null,
    is_match bool not null,
    threshold float not null,
    created_at float not null
);

create or replace view latest_global_object_ids as
    select
        *
    from intervideo_object_ids
    where
        created_at = (select max(created_at) from intervideo_object_ids)
        and is_match;

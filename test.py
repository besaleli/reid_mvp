from ultralytics import YOLO
import cv2
import torch
import pandas as pd
from torchvision import transforms
from torch.nn.functional import normalize
import torchreid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, deque

# Load YOLOv8 model for tracking people
yolo_model = YOLO("yolo11n.pt")
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"RUNNING ON {device}")

# Load OSNet model
osnet_model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    loss='softmax',
    pretrained=True
).to(device).eval()

# Transform for OSNet input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class IDStabilizer:
    def __init__(self, similarity_threshold=0.75, window_size=5):
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        
        # Stable ID management
        self.stable_id_counter = 1
        self.yolo_to_stable = {}  # yolo_id -> stable_id
        self.stable_embeddings = {}  # stable_id -> list of embeddings
        
        # Temporal smoothing
        self.track_history = defaultdict(lambda: deque(maxlen=window_size))
        
        # Missing track handling
        self.missing_tracks = {}  # stable_id -> frames_missing
        self.max_missing_frames = 10
        
    def get_stable_id(self, yolo_id, embedding, frame_id):
        """Get stable ID for a YOLO track ID"""
        
        # Validate embedding (prevent zero-norm vectors)
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm < 1e-8:
            print(f"Warning: Near-zero embedding norm ({embedding_norm:.2e}) for YOLO ID {yolo_id}")
            return None
        
        # Normalize embedding to prevent cosine similarity issues
        embedding = embedding / embedding_norm
        
        # If we've seen this YOLO ID before, check if it's still the same person
        if yolo_id in self.yolo_to_stable:
            stable_id = self.yolo_to_stable[yolo_id]
            
            # Verify it's still the same person
            if stable_id in self.stable_embeddings:
                stored_embeddings = np.array(self.stable_embeddings[stable_id])
                
                # Safe cosine similarity calculation
                try:
                    similarities = cosine_similarity([embedding], stored_embeddings)[0]
                    avg_similarity = np.mean(similarities)
                    
                    if not np.isfinite(avg_similarity):
                        print(f"Warning: Non-finite similarity for YOLO ID {yolo_id}")
                        avg_similarity = 0.0
                        
                except (ValueError, ZeroDivisionError) as e:
                    print(f"Error computing similarity for YOLO ID {yolo_id}: {e}")
                    avg_similarity = 0.0
                
                if avg_similarity > self.similarity_threshold:
                    # Same person, update embeddings
                    self.stable_embeddings[stable_id].append(embedding)
                    if len(self.stable_embeddings[stable_id]) > 20:  # Keep recent 20
                        self.stable_embeddings[stable_id].pop(0)
                    
                    # Reset missing counter
                    if stable_id in self.missing_tracks:
                        del self.missing_tracks[stable_id]
                    
                    return stable_id
                else:
                    # ID switch detected, reassign
                    print(f"ID switch detected for YOLO ID {yolo_id} (similarity: {avg_similarity:.3f})")
                    del self.yolo_to_stable[yolo_id]
        
        # Find best matching existing person or create new one
        best_stable_id = None
        best_similarity = 0
        
        for stable_id, stored_embeddings in self.stable_embeddings.items():
            if stable_id in self.missing_tracks:  # Skip recently missing tracks
                continue
                
            stored_embeddings_array = np.array(stored_embeddings)
            
            # Safe similarity computation
            try:
                similarities = cosine_similarity([embedding], stored_embeddings_array)[0]
                avg_similarity = np.mean(similarities)
                
                if not np.isfinite(avg_similarity):
                    continue
                    
            except (ValueError, ZeroDivisionError):
                continue
            
            if avg_similarity > best_similarity and avg_similarity > self.similarity_threshold:
                best_stable_id = stable_id
                best_similarity = avg_similarity
        
        if best_stable_id:
            # Reassign to existing person
            self.yolo_to_stable[yolo_id] = best_stable_id
            self.stable_embeddings[best_stable_id].append(embedding)
            print(f"Reassigned YOLO ID {yolo_id} to stable ID {best_stable_id} (similarity: {best_similarity:.3f})")
            return best_stable_id
        else:
            # Create new person
            new_stable_id = self.stable_id_counter
            self.stable_id_counter += 1
            self.yolo_to_stable[yolo_id] = new_stable_id
            self.stable_embeddings[new_stable_id] = [embedding]
            print(f"Created new stable ID {new_stable_id} for YOLO ID {yolo_id}")
            return new_stable_id
    
    def handle_missing_tracks(self, current_yolo_ids):
        """Handle tracks that disappeared (likely due to occlusion)"""
        active_stable_ids = set(self.yolo_to_stable.values())
        
        # Mark missing stable IDs
        for stable_id in list(self.stable_embeddings.keys()):
            if stable_id not in active_stable_ids:
                if stable_id not in self.missing_tracks:
                    self.missing_tracks[stable_id] = 0
                self.missing_tracks[stable_id] += 1
                
                # Remove if missing too long
                if self.missing_tracks[stable_id] > self.max_missing_frames:
                    print(f"Removing stable ID {stable_id} (missing too long)")
                    del self.stable_embeddings[stable_id]
                    del self.missing_tracks[stable_id]
                    # Remove from yolo mapping if exists
                    self.yolo_to_stable = {k: v for k, v in self.yolo_to_stable.items() if v != stable_id}

def filter_detections(result, min_area=2000, min_confidence=0.4):
    """Filter out low-quality detections that might cause ID switches"""
    if not result.boxes or len(result.boxes) == 0:
        return None
    
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
    
    # Validate box dimensions to prevent divide by zero
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    
    # Filter out degenerate boxes (zero or negative width/height)
    valid_boxes_mask = (widths > 1) & (heights > 1)
    
    if not np.any(valid_boxes_mask):
        return None
    
    # Apply box validation filter first
    boxes = boxes[valid_boxes_mask]
    confidences = confidences[valid_boxes_mask]
    if ids is not None:
        ids = ids[valid_boxes_mask]
    
    # Recalculate with validated boxes
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Filter by area and confidence
    valid_mask = (areas > min_area) & (confidences > min_confidence)
    
    if not np.any(valid_mask):
        return None
    
    # Create filtered result
    filtered_boxes = boxes[valid_mask]
    filtered_confidences = confidences[valid_mask]
    filtered_ids = ids[valid_mask] if ids is not None else None
    
    return {
        'boxes': filtered_boxes,
        'confidences': filtered_confidences,
        'ids': filtered_ids,
        'orig_img': result.orig_img
    }

# Initialize stabilizer
stabilizer = IDStabilizer()

# Video stream
cap = cv2.VideoCapture("data/video_1.mp4")
data = []
frame_count = -1

while cap.isOpened():
    frame_count += 1
    ret, frame = cap.read()
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    if not ret:
        break

    # YOLO tracking with optimized BoTSORT settings
    results = yolo_model.track(
        frame, 
        persist=True, 
        classes=[0],  # Only 'person'
        conf=0.25,    # Even lower confidence for partial occlusions
        iou=0.5,      # Lower IoU - let BoTSORT handle association
        tracker="botsort.yaml"  # Use custom config
    )

    if not results or not results[0].boxes:
        stabilizer.handle_missing_tracks([])
        continue

    result = results[0]
    
    # Filter detections
    filtered_result = filter_detections(result, min_area=1500, min_confidence=0.3)
    if not filtered_result:
        stabilizer.handle_missing_tracks([])
        continue
    
    boxes = filtered_result['boxes']
    confidences = filtered_result['confidences']
    yolo_ids = filtered_result['ids']
    
    if yolo_ids is None:
        stabilizer.handle_missing_tracks([])
        continue
    
    # Extract crops and get embeddings
    crops = []
    valid_boxes = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        # Validate crop dimensions
        if x2 <= x1 or y2 <= y1:
            print(f"Skipping invalid box: ({x1},{y1},{x2},{y2})")
            continue
            
        # Ensure coordinates are within image bounds
        h, w = filtered_result['orig_img'].shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Final dimension check
        if x2 <= x1 or y2 <= y1:
            print(f"Skipping out-of-bounds box: ({x1},{y1},{x2},{y2})")
            continue
            
        crop_bgr = filtered_result['orig_img'][y1:y2, x1:x2]
        
        # Check if crop is empty or too small
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
            print(f"Skipping tiny/empty crop: {crop_bgr.shape}")
            continue
            
        try:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            transformed_crop = transform(crop_rgb)
            crops.append(transformed_crop)
            valid_boxes.append(i)  # Track which box this crop corresponds to
        except Exception as e:
            print(f"Error processing crop {i}: {e}")
            continue

    if not crops:
        stabilizer.handle_missing_tracks([])
        continue

    # Get OSNet embeddings
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        embeddings = osnet_model(batch)
        embeddings = normalize(embeddings, dim=1, eps=1e-8).cpu().numpy()  # Add eps to prevent division by zero

    # Handle missing tracks
    current_yolo_ids = [int(yolo_ids[i]) for i in valid_boxes]
    stabilizer.handle_missing_tracks(current_yolo_ids)
    
    # Process each valid detection
    for i, box_idx in enumerate(valid_boxes):
        if i >= len(embeddings):  # Safety check
            break
            
        yolo_id = int(yolo_ids[box_idx])
        confidence_score = confidences[box_idx]
        embedding = embeddings[i]
        
        # Get stable ID (with error handling)
        stable_id = stabilizer.get_stable_id(yolo_id, embedding, frame_count)
        
        if stable_id is not None:  # Only add if we got a valid stable ID
            data.append({
                "confidence_score": confidence_score,
                "yolo_id": yolo_id,
                "stable_id": stable_id,
                "osnet_embedding": embedding,
                "timestamp": timestamp,
                "frame_id": frame_count
            })

    # Visualization
    annotated_frame = result.orig_img.copy()
    for i, box_idx in enumerate(valid_boxes):
        if i < len(embeddings):
            yolo_id = int(yolo_ids[box_idx])
            stable_id = stabilizer.yolo_to_stable.get(yolo_id, "Unknown")
            box = boxes[box_idx]
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw labels
            label = f"Y:{yolo_id} S:{stable_id}"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Create DataFrame
df = pd.DataFrame(data)
print(f"\nProcessed {len(df)} detections")
print(f"Unique YOLO IDs: {df['yolo_id'].nunique()}")
print(f"Unique Stable IDs: {df['stable_id'].nunique()}")

# Show ID mapping summary
print("\nFinal ID mappings:")
for yolo_id, stable_id in stabilizer.yolo_to_stable.items():
    count = len(df[(df['yolo_id'] == yolo_id) & (df['stable_id'] == stable_id)])
    print(f"YOLO ID {yolo_id} -> Stable ID {stable_id} ({count} detections)")

# Save results
df.to_pickle("tracking_results.pkl")
print("\nResults saved to tracking_results.pkl")

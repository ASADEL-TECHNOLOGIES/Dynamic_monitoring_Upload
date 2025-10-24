# main.py
import cv2
import time
import os
from ultralytics import YOLO
from db import load_config, get_db_connection 
from roi_handler import select_rois_interactive, load_rois_from_file
import datetime

def run_camera(cam_name=None, cam_source=None):

    #-----------LOAD CONFIG----------
    config = load_config()
    model_path = config["system"]["model"]
    video_path = cam_source or config["cameras"][0]["source_path"]
    interval = config["system"]["interval_second"]
    db_config = config["database"]
    class_name_map = config["classes"]
    conf = config["system"]["conf"]
    frame_skip = config["system"]["frame_skip"]
    use_gpu = config["system"]["use_gpu"]

    #---------------Db connection------------
    db = get_db_connection(db_config)
    cursor = db.cursor()

    # ---------MODEL-----------
    if use_gpu:
        model = YOLO(model_path).to("cuda")
        print("Using GPU for inference")
    else :
        model = YOLO(model_path).to("cpu")
        print("Using CPU for inference")

    # ---------- ROI FILE ----------
    model_name = os.path.basename(model_path).split('.')[0] 
    cam_name = cam_name or "default_cam"

    roi_dir = os.path.join(os.getcwd(), "rois", model_name)
    os.makedirs(roi_dir, exist_ok=True)

    roi_file = os.path.join(roi_dir, f"{cam_name}.json")
    print("ROIs will be saved at:", roi_file)

    # Load existing ROIs (no input prompt)
    rois = select_rois_interactive(video_path, roi_file, class_name_map)

    #----------------YOLO TRACKING-----------
    results = model.track(source=video_path, conf=conf, iou=0.5, show=False, stream=True, verbose=False, vid_stride=frame_skip+1)

    unique_ids_seen = {cls_id: set() for cls_id in rois.keys()}
    last_insert_time = time.time()
    prev_counts = {cls_id: 0 for cls_id in rois.keys()}



    for result in results:
        frame = result.orig_img.copy()
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = class_name_map.get(str(cls_id), f"Class {cls_id}")
            track_id = int(box.id[0].item()) if box.id is not None else None
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            if cls_id in rois:
                x, y, w, h = rois[cls_id]
                cx, cy = (xyxy[0] + xyxy[2]) // 2, (xyxy[1] + xyxy[3]) // 2
                if x <= cx <= x + w and y <= cy <= y + h:
                    if track_id is not None:
                        unique_ids_seen[cls_id].add(track_id)

                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"{cls_name} | ID={track_id}",
                                (xyxy[0], xyxy[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw ROIs
        for cls_id, (x, y, w, h) in rois.items():
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"ROI-{cls_id}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        y_offset = 40
        for cls_id, ids in unique_ids_seen.items():
            cls_name = class_name_map.get(str(cls_id), f"Class {cls_id}")
            cv2.putText(frame, f"{cls_name} Count: {len(ids)}",
                        (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            y_offset += 40

        # Insert to DB per interval
        if time.time() - last_insert_time > interval:
            for cid, ids in unique_ids_seen.items():
                total = len(ids)
                new_count = total - prev_counts[cid]
                prev_counts[cid] = total
                detected = bool(new_count)
                cls_name = class_name_map.get(str(cid), f"Class {cid}")

                cursor.execute("""
                    INSERT INTO entries (class_id, name, count, detected, camera_name)
                    VALUES (%s, %s, %s, %s, %s)
                """, (cid, cls_name, new_count, detected, cam_name))
                db.commit()

            last_insert_time = time.time()
            print(f"✅ DB updated at {time.strftime('%H:%M:%S')}")

        cv2.imshow(f"ROI-Based Tracking - {cam_name}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera()

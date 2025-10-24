import cv2
from ultralytics import YOLO
import json
import time
import mysql.connector

#--------------load config--------------
with open("/home/asadel/Desktop/Cylinder_Project/main/config.json" ,"r") as f:
    config = json.load(f)

model_path = config["system"]["model"]
video_path = config["system"]["source_path"]
interval = config["system"]["interval_second"]

db_config = config["database"]

#-------------load database config------------

db = mysql.connector.connect(
    host = db_config["host"],
    user = db_config["username"],
    password = db_config["password"],
    port = db_config["port"],
    database = db_config["db"]
)

cursor = db.cursor()

#-----------------custom names----------------
class_name_map = {
    0: "cap shot",
    1: "cap",
    2: "cylinder",
    3: "wheel chock"
}

#-------------load model-----------------

model = YOLO(model_path)


# ---------- STEP 1: ROI SELECTION ----------
print("\n Select ROIs for classes (auto IDs start from 0)")
print(" Press SPACE or ENTER to confirm ROI, or 'c'/'ESC' to finish.\n")

# Grab one frame from video for ROI drawing
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError(" Failed to read frame for ROI selection!")

rois = {}
class_counter = 0

while True:
    r = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    if r == (0, 0, 0, 0):
        print("No ROI selected, exiting ROI setup.")
        break

    rois[class_counter] = r
    print(f"✅ ROI saved for class {class_counter}: {r}")
    class_counter += 1

    print("Press ESC to stop adding more ROIs or ENTER to continue...")
    key = cv2.waitKey(0)
    if key in [27, ord('c')]:  # ESC or 'c' key
        print("✅ Finished ROI selection.\n")
        break

cv2.destroyWindow("Select ROI")

print("📋 Final ROIs:")
for cls_id, roi in rois.items():
    print(f"Class {cls_id}: {roi}")
print("-" * 40)


# ---------- STEP 2: YOLO TRACKING ----------
results = model.track(source=video_path, conf=0.5, iou=0.5, show=False, stream=True, verbose= False)

# To keep count of unique objects (like cylinders)
unique_ids_seen = {cls_id: set() for cls_id in rois.keys()}
last_insert_time = time.time()
prev_counts = {cls_id: 0 for cls_id in rois.keys()}

for result in results:
    frame = result.orig_img.copy()
    boxes = result.boxes
    current_other_ids = set()

    for box in boxes:
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        conf = float(box.conf[0].item())
        track_id = int(box.id[0].item()) if box.id is not None else None
        xyxy = box.xyxy[0].cpu().numpy().astype(int)

        # Check if ROI exists for this class
        if cls_id in rois:
            x, y, w, h = rois[cls_id]
            roi_x2, roi_y2 = x + w, y + h

            # Get center of box
            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int((xyxy[1] + xyxy[3]) / 2)

            # Check if box center lies inside ROI
            if x <= cx <= roi_x2 and y <= cy <= roi_y2:
                if track_id is not None:
                    unique_ids_seen[cls_id].add(track_id)

                # Draw bounding box
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"Class {cls_id} | ID={track_id}",
                            (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            current_other_ids.add(cls_id)
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 255), 2)
            cv2.putText(frame, f"Detected Outside ROI: {cls_id}",
                        (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Overlay ROI boxes
    for cls_id, (x, y, w, h) in rois.items():
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"ROI-Class {cls_id}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display counts
    y_offset = 40
    for cls_id, ids in unique_ids_seen.items():
        cv2.putText(frame, f"Class {cls_id} Count: {len(ids)}",
                    (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        y_offset += 40


    #-------------------insert into database here (Per Interval)--------------
    current_time = time.time()
    if current_time - last_insert_time > interval:
        
        for cid, ids in unique_ids_seen.items():
            total_count = len(ids)
            interval_count = total_count - prev_counts[cid] # New items only 
            prev_counts[cid] = total_count

            detected = True if interval_count > 0 else False
            class_name = class_name_map.get(cid, f"Class {cid}")

            insert_query = """ 
                INSERT INTO entries(class_id, name, count, detected)
                VALUES (%s, %s, %s, %s)
            """
            values = (cid, class_name, interval_count, detected)
            cursor.execute(insert_query, values)
            db.commit()

        last_insert_time = current_time
        print(f"✅ Data inserted to DB at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Show frame
    cv2.imshow("ROI-based Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

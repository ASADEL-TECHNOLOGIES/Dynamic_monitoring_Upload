import cv2
from ultralytics import YOLO

model = YOLO("/home/asadel/Desktop/Cylinder_Project/model/cylinder_best .pt")

results = model.track(
    source="/home/asadel/Desktop/Cylinder_Project/videos/cylinder.mp4",
    conf=0.5,
    iou=0.5,
    show=False,  # we’ll use our own cv2.imshow
    stream=True
)

cylinder_ids_seen = set()   #unique cylinder ids only

for result in results:
    frame = result.orig_img.copy()
    boxes = result.boxes

    current_other_ids = set()

    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        track_id = int(box.id[0].item()) if box.id is not None else None

        xyxy = box.xyxy[0].cpu().numpy().astype(int)    #xyxy coordinates of bounding box as a tensor, changed to numpy array

        if cls_id == 2:
            if track_id is not None:
                # count unique cylinders
                cylinder_ids_seen.add(track_id)

            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"Cylinder-{track_id}", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            current_other_ids.add(cls_id)
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 255, 0), 2)
            cv2.putText(frame, f"Detected ID={cls_id}", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            

    # Overlay info
    cv2.putText(frame, f"Unique Cylinders: {len(cylinder_ids_seen)}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    if current_other_ids:
        cv2.putText(frame, f"Detected IDs: {', '.join(map(str, sorted(current_other_ids)))}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Terminal Output
    print(f"✅ Total Unique Cylinders (ID=2): {len(cylinder_ids_seen)}")
    if current_other_ids:
        print(f"⚠️ Detected Other IDs: {sorted(current_other_ids)}")


    cv2.imshow("Cylinder Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
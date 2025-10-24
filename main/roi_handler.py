import cv2
import os
import json

def load_rois_from_file(roi_file):
    if not os.path.exists(roi_file):
        return {}
    with open(roi_file, "r") as f:
        data = json.load(f)
    print(f" Loaded ROIs from {roi_file}")
    return {int(k): tuple(v) for k,v in data.items()}


def save_rois_to_file(rois, roi_file):
    if not rois:
        print(" No ROIs to save!")
        return
    os.makedirs(os.path.dirname(roi_file), exist_ok=True)
    with open(roi_file, "w") as f:
        json.dump({str(k): v for k, v in rois.items()}, f, indent=2)
    print(f"✅ ROIs saved to {roi_file}")


def select_rois_interactive(video_path, roi_file, class_name_map, redraw=False):
    if not redraw and os.path.exists(roi_file):
        print(f" Loaded existing ROIs from {roi_file}")
        return load_rois_from_file(roi_file)

        
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError("Failed to read frame for ROI selection")
    
    rois = {}
    print("\n Select ROI for configured classes (Press ENTER to confirm, ESC to skip)\n)")
    
    # Draw ROI for each configured class
    for cls_id, cls_name in class_name_map.items():
        print(f"Draw ROI for class ID {cls_id} → '{cls_name}'")
        r = cv2.selectROI(f"ROI for {cls_name}", frame, showCrosshair=True, fromCenter=False)

        if r == (0, 0, 0, 0):
            print(f"⚠ Skipped class '{cls_name}' (no ROI selected).")
            continue

        rois[int(cls_id)] = [int(v) for v in r]
        print(f"✅ ROI saved for class '{cls_name}' (ID {cls_id}): {r}")
        cv2.destroyWindow(f"ROI for {cls_name}")

    cv2.destroyAllWindows()

    # Save all ROIs
    if rois:
        save_rois_to_file(rois, roi_file)
    else:
        print(" No ROIs selected, nothing saved.")

    return rois
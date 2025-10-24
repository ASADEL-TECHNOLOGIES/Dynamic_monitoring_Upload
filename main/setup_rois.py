import os 
import json
from roi_handler import select_rois_interactive, save_rois_to_file, load_rois_from_file
from db import load_config


#load config
config = load_config()
model_path = config["system"]["model"]
class_name_map = config['classes']
cameras = config["cameras"]

model_name = os.path.basename(model_path).split('.')[0]
roi_dir = os.path.join(os.getcwd(), "rois", model_name)
os.makedirs(roi_dir, exist_ok=True)

for cam in cameras:
    cam_name = cam["name"]
    roi_file = os.path.join(roi_dir, f"{cam_name}.json")
    
    # Check if ROI file exists and ask for redraw
    redraw = False
    if os.path.exists(roi_file):
        response = input(f"ROI file exists for {cam_name}. Do you want to redraw? (y/n): ")
        if response.lower() in ['y', 'yes']:
            redraw = True
    
    print(f"\n--- Setting up ROIs for camera: {cam_name} ---")
    rois = select_rois_interactive(cam["source_path"], roi_file, class_name_map, redraw)
    print(f"✅ ROIs saved for {cam_name} at {roi_file}")
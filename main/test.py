# start_all_threading.py
import json
import threading
from main import run_camera
from db import load_config

if __name__ == "__main__":
    config = load_config()
    cameras = config['cameras']

    threads = []

    for cam in cameras:
        t = threading.Thread(target=run_camera, args=(cam['name'], cam['source_path']))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

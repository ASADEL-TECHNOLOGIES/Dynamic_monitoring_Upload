# start_all.py
import json
import os
import multiprocessing
import threading
from main import run_camera
from db import load_config

def run_with_threading(cameras):
    threads = []

    for cam in cameras:
        t = threading.Thread(target=run_camera, args=(cam['name'], cam['source_path']))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

def run_with_multiprocessing(cameras):
    processes = []

    for cam in cameras:
        p = multiprocessing.Process(target=run_camera, args=(cam['name'], cam['source_path']))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    config = load_config()
    cameras = config['cameras']
    run_mode = config["system"]["run_mode"]

    print(f" Running camera in '{run_mode}' mode ")

    if run_mode == "threading":
        run_with_threading(cameras)
    elif run_mode =="multiprocessing":
        run_with_multiprocessing(cameras)    
    else:
        print("Invalid run_mode in config. Use 'threading' or 'multiprocessing' ")

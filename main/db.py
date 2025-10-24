import mysql.connector
import json
import os

def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'main', 'config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_db_connection(db_config):
    try:
        return mysql.connector.connect(
            host=db_config["host"],
            user=db_config["username"],
            password=db_config["password"],
            port=db_config["port"],
            database=db_config["db"]
        )
    except mysql.connector.Error as err:
        raise RuntimeError(f"❌ Database connection failed: {err}")

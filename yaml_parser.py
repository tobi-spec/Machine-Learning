import yaml
import json
import logging

def get_hyperparameters(path: str):
    try:
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"No file in {path}")
    except IOError:
        logging.error(f"IOError: An I/O error occurred for path {path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


def get_api_key():
    try:
        with open("./api-key.yaml", 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error("No file with name 'api-key.yaml' in directory")
    except IOError:
        logging.error("IOError: An I/O error occurred.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


def write_to_txt(text: dict, path: str):
    try:
        with open(path, 'w') as file:
            json.dump(text, file)
    except IOError as e:
        logging.error(f"An I/O error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
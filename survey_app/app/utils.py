import json
import os
import logging


def load_json(filepath):
    with open(filepath, 'r') as fin:
        return json.loads(fin.read())


def get_logger():
    os.makedirs('logs', exist_ok=True)
    logpath = 'logs/log.log'
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler(logpath)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

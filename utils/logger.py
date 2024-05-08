import logging
import datetime
import os

def construct_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    date = str(datetime.datetime.now().strftime("%m%d%H%M"))
    fh = logging.FileHandler(os.path.join(save_dir, f"log-{date}.txt"))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def logging_information(logger, info):
    #if is_main_process():
    logger.info(info)
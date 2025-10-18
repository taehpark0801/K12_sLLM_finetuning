#-*- coding:utf-8 -*-

import logging

def getLogger(
    logging_path,
    logging_name="noName"
):

    log_format = '[%(levelname)s|%(funcName)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s'
    logger = logging.getLogger(logging_name)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(logging_path)
    streamHandler = logging.StreamHandler()
    formatter = logging.Formatter(log_format)
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    return logger

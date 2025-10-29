import logging


def get_logger(name: str, logging_level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s.%(msecs)03d] %(levelname)s | %(name)s | %(message)s",
            datefmt="%d-%b-%Y %H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.setLevel(logging_level)
    return logger

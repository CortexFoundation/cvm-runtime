import logging
from mrt.common.log import (
    LOG_LEVELS, ColorFormatter, FilterList, name2level
)

def log_init(log_level, streamer):
    assert log_level in LOG_LEVELS
    logging.basicConfig(level=log_level, stream=streamer)
    formatter = ColorFormatter(
        fmt="[ %(asctime)s %(name)10s %(levelname)5s ] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    log_filter = FilterList(log_level=log_level, default=False)
    for handler in logging.root.handlers:
        handler.addFilter(log_filter)
        handler.setFormatter(formatter)

def get_logger(verbosity, streamer):
    log_init(name2level(verbosity.upper()), streamer)
    logger = logging.getLogger("log.main")
    return logger

from datetime import datetime
from typing import List

import logging

TRACE = logging.DEBUG // 2
DEBUG = logging.DEBUG
INFO = logging.INFO
WARN = logging.WARNING
ERROR = logging.ERROR
FATAL = logging.CRITICAL

logging.addLevelName(TRACE, "TRACE")
logging.addLevelName(DEBUG, "DEBUG")
logging.addLevelName(INFO,  "INFO")
logging.addLevelName(WARN,  "WARN")
logging.addLevelName(ERROR, "ERROR")
logging.addLevelName(FATAL, "FATAL")

LOG_LEVELS = [TRACE, DEBUG, INFO, WARN, ERROR, FATAL]
LOG_NAMES = [logging.getLevelName(l).strip() for l in LOG_LEVELS]

def level2name(log_level):
    assert log_level in LOG_LEVELS
    return LOG_NAMES[LOG_LEVELS.index(log_level)]

def name2level(log_name):
    assert log_name in LOG_NAMES
    return LOG_LEVELS[LOG_NAMES.index(log_name)]


class ColorFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super(ColorFormatter, self).__init__(fmt, datefmt, style)

        self._colors = {
            "TRACE": "\033[38;5;111m",
            "DEBUG": "\033[38;5;111m",
            "INFO": "\033[38;5;47m",
            "WARN": "\033[38;5;178m",
            "ERROR": "\033[38;5;196m",
            "FATAL": "\033[30;48;5;196m",
        }
        self._default = "\033[38;5;15m"
        self._reset = "\033[0m"

    def format(self, record):
        message = super(ColorFormatter, self).format(record)
        log_color = self._colors.get(record.levelname, self._default)
        message = log_color + message + self._reset
        return message

class FilterList(logging.Filter):
    """ Filter with logging module

        Filter rules as below:
            {allow|disable log name} > level no > keywords >
            {inheritance from parent log name} > by default filter
    """
    def __init__(self, default=False, allows=[], disables=[],
            keywords=[], log_level=logging.INFO):
        self.rules = {}
        self._internal_filter_rule = "_internal_filter_rule"
        self.log_level = log_level
        self.keywords = keywords

        self.rules[self._internal_filter_rule] = default
        for name in allows:
            splits = name.split(".")
            rules = self.rules
            for split in splits:
                if split not in rules:
                    rules[split] = {}
                rules = rules[split]

            rules[self._internal_filter_rule] = True

        for name in disables:
            splits = name.split(".")
            rules = self.rules
            for split in splits:
                if split not in rules:
                    rules[split] = {}
                rules = rules[split]

            rules[self._internal_filter_rule] = False

    def filter(self, record):
        rules = self.rules
        rv = rules[self._internal_filter_rule]

        splits = record.name.split(".")
        for split in splits:
            if split in rules:
                rules = rules[split]
                if self._internal_filter_rule in rules:
                    rv = rules[self._internal_filter_rule]
            else:
                if record.levelno >= self.log_level:
                    return True

                for keyword in self.keywords:
                    if keyword in record.getMessage():
                        return True
                return rv
        return rv

def Init(log_level):
    assert log_level in LOG_LEVELS
    logging.basicConfig(level=log_level)
    formatter = ColorFormatter(
            fmt="[ %(asctime)s %(name)10s %(levelname)5s ] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")

    log_filter = FilterList(
                log_level=log_level,
                default=False)
    for handler in logging.root.handlers:
        handler.addFilter(log_filter)
        handler.setFormatter(formatter)

if __name__ == "__main__":
    from . import cmd

    @cmd.module("log", as_main=True,
        help="log test module", permission=cmd.PRIVATE)
    def test_main(args):
        Init(args)
        logging.debug("test")
        logging.info("test")
        logging.warning("test")
        logging.error("test")

    cmd.Run()

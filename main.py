import sys
from os import path

# set up dependencies
__ROOT__ = path.dirname(path.realpath(__file__))
sys.path.insert(0, path.join(__ROOT__, "python"))

import logging

from mrt.common import cmd, log, thread

LOG_MSG = ",".join(["{}:{}".format(l, n) \
    for l, n in zip(log.LOG_LEVELS, log.LOG_NAMES)])

@cmd.option("-v", "--verbosity", metavar="LEVEL",
            choices=log.LOG_NAMES, default=log.level2name(log.DEBUG),
            help="log verbosity to pring information, " + \
                "available options: {}".format(log.LOG_NAMES) + \
                " by default {}".format(log.level2name(log.DEBUG)))
@cmd.global_options()
def global_func(args):
    log.Init(log.name2level(args.verbosity))

@cmd.module("", as_main=True,
            description="""
CVM Python Tool
""")
def cvm_main(args):
    print("null")
    thread.start_services(args)

if __name__ == "__main__":
    logger = logging.getLogger("main")
    cmd.Run()

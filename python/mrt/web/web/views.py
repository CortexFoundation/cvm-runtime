from queue import Queue, Empty
from threading import Thread, current_thread
import time
import sys
import os
import logging

from django.http.response import StreamingHttpResponse
from django.shortcuts import render

from mrt.V3.utils import get_cfg_defaults
from mrt.V3.prepare import prepare
from mrt.V3.calibrate import calibrate
from mrt.V3.quantize import quantize
from mrt.V3.evaluate import evaluate
from mrt.V3.mrt_compile import mrt_compile
from .log import get_logger

class Printer:
    def __init__(self):
        self.queues = {}

    def write(self, value):
        queue = self.queues.get(current_thread().name)
        if queue:
            queue.put(value)
        else:
            sys.__stdout__.write(value)

    def flush(self):
        pass

    def register(self, thread):
        queue = Queue()
        self.queues[thread.name] = queue
        return queue

    def clean(self, thread):
        del self.queues[thread.name]

printer = Printer()
sys.stdout = printer


class Streamer:
    def __init__(self, target, args):
        self.thread = Thread(target=target, args=args)
        self.queue = printer.register(self.thread)

    def start(self):
        self.thread.start()
        # print('This should be stdout')
        while self.thread.is_alive():
            try:
                item = self.queue.get_nowait()
                yield f'{item}<br>'
            except Empty:
                pass
        yield '<br>***End***<br>'
        printer.clean(self.thread)

mrt_web_tmp_dir = os.path.expanduser("~/.mrt_web")
os.makedirs(mrt_web_tmp_dir, exist_ok=True)

def get_cfg(yaml_file, attr):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(yaml_file)
    cfg.freeze()
    cm_cfg = cfg.COMMON
    pass_cfg = getattr(cfg, attr)
    return cm_cfg, pass_cfg

def mrt_prepare_log(request):
    yaml_file = os.path.expanduser("~/mrt_yaml_root/alexnet.yaml")
    cm_cfg, pass_cfg = get_cfg(yaml_file, "PREPARE")
    logger = get_logger(cm_cfg.VERBOSITY, printer)
    streamer = Streamer(prepare, (cm_cfg, pass_cfg, logger))
    return StreamingHttpResponse(streamer.start())

def mrt_calibrate_log(request):
    yaml_file = os.path.expanduser("~/mrt_yaml_root/alexnet.yaml")
    cm_cfg, pass_cfg = get_cfg(yaml_file, "CALIBRATE")
    logger = get_logger(cm_cfg.VERBOSITY, printer)
    streamer = Streamer(calibrate, (cm_cfg, pass_cfg, logger))
    return StreamingHttpResponse(streamer.start())

def mrt_quantize_log(request):
    yaml_file = os.path.expanduser("~/mrt_yaml_root/alexnet.yaml")
    cm_cfg, pass_cfg = get_cfg(yaml_file, "QUANTIZE")
    logger = get_logger(cm_cfg.VERBOSITY, printer)
    streamer = Streamer(quantize, (cm_cfg, pass_cfg, logger))
    return StreamingHttpResponse(streamer.start())

def mrt_evaluate_log(request):
    yaml_file = os.path.expanduser("~/mrt_yaml_root/alexnet.yaml")
    cm_cfg, pass_cfg = get_cfg(yaml_file, "EVALUATE")
    logger = get_logger(cm_cfg.VERBOSITY, printer)
    streamer = Streamer(evaluate, (cm_cfg, pass_cfg, logger))
    return StreamingHttpResponse(streamer.start())

def mrt_compile_log(request):
    yaml_file = os.path.expanduser("~/mrt_yaml_root/alexnet.yaml")
    cm_cfg, pass_cfg = get_cfg(yaml_file, "COMPILE")
    logger = get_logger(cm_cfg.VERBOSITY, printer)
    streamer = Streamer(mrt_compile, (cm_cfg, pass_cfg, logger))
    return StreamingHttpResponse(streamer.start())

def room(request, room_name):
    return render(request, "room.html", {"room_name": room_name})

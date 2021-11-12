import json
import os
from channels.generic.websocket import WebsocketConsumer

from .log import get_logger
from .streamer import Streamer, printer
from mrt.V3.utils import merge_cfg
from mrt.V3.execute import run

class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        yaml_file = text_data_json['yaml_file']
        cfg = merge_cfg(yaml_file)
        pass_name = text_data_json['pass_name']
        logger = get_logger(cfg.COMMON.VERBOSITY, printer)
        my_streamer = Streamer(run, (cfg, pass_name, logger))
        for message in my_streamer.start():
            self.send(text_data=json.dumps({'message': message}))

import json
import yaml
import os
from os import path
from channels.generic.websocket import WebsocketConsumer

from .log import get_logger
from .streamer import Streamer, printer
from mrt.V3.utils import merge_cfg, revise_cfg, get_cfg_defaults
from mrt.V3.execute import run
from .protocol import type_cast

tmp_dir = path.expanduser("~/mrt_tmp")
os.makedirs(tmp_dir, exist_ok=True)
activation_flag = "___activate_executor___"


class MRTExecuteConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        json_from_js = json.loads(text_data)
        json_data = {}
        for stage, stage_data in json_from_js.items():
            sub_type_cast = type_cast[stage]
            sub_json_data = {}
            for attr, data in stage_data.items():
                if data == '':
                    continue
                if attr in sub_type_cast:
                    cast_func = sub_type_cast[attr]
                    data = cast_func(data)
                sub_json_data[attr] = data
            json_data[stage] = sub_json_data
        yaml_str = yaml.dump(json_data)
        tmp_yaml_file = path.join(tmp_dir, "tmp.yaml")
        with open(tmp_yaml_file, "w") as f:
            f.write(yaml_str)
        cfg = get_cfg_defaults()
        cfg.merge_from_file(tmp_yaml_file)
        logger = get_logger(cfg.COMMON.VERBOSITY, printer)
        # revise_cfg(cfg, "COMMON", "PASS_NAME", pass_name)
        my_streamer = Streamer(run, (cfg, logger))
        for message in my_streamer.start():
            self.send(text_data=json.dumps({'message': message}))
        self.send(text_data=json.dumps({'message': activation_flag}))


class YAMLInitConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        cfg = get_cfg_defaults()
        self.send(text_data=json.dumps(cfg))


class YAMLUpdateConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        yaml_file = text_data_json['yaml_file']
        cfg = merge_cfg(yaml_file)
        self.send(text_data=json.dumps(cfg))


class YAMLClearConsumer(YAMLInitConsumer):
    pass


class YAMLCollectConsumer(YAMLInitConsumer):
    pass

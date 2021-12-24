import json
import yaml
import os
from os import path

from channels.generic.websocket import WebsocketConsumer

from mrt.V3.utils import merge_cfg, get_cfg_defaults
from mrt.V3.execute import run
from rpc.service import mrt_execute, mrt_submit
from rpc.utils import stringify_cfg
from .protocol import type_cast


class MRTExecuteConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        json_from_js = json.loads(text_data)
        json_data = {}
        ref_cfg = get_cfg_defaults()
        for stage, stage_data in json_from_js['yaml'].items():
            sub_type_cast = type_cast[stage]
            sub_json_data = {}
            stage_ref_data = getattr(ref_cfg, stage)
            for attr, data in stage_data.items():
                if data == '':
                    data = getattr(stage_ref_data, attr)
                elif attr in sub_type_cast:
                    cast_func = sub_type_cast[attr]
                    data = cast_func(data)
                sub_json_data[attr] = data
            json_data[stage] = sub_json_data
        yaml_file_str = yaml.dump(json_data)
        host_addr = json_from_js['host']
        for message in mrt_execute(yaml_file_str, host_addr=host_addr):
            self.send(text_data=json.dumps({'message': message}))
        self.send(
            text_data=json.dumps({'activate': None}))


class ModelSubmitConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        json_from_js = json.loads(text_data)
        json_data = {}
        host_addr = json_from_js['host']
        src_sym_file = json_from_js['symbol']
        src_prm_file = json_from_js['params']
        dst_model_dir = json_from_js['dst']
        cnt = 0
        for message in mrt_submit(
            src_sym_file, src_prm_file, dst_model_dir,
            host_addr=host_addr):
            cnt += 1
            dct = {'message': message}
            if cnt == 1:
                dct['first'] = 1
            self.send(text_data=json.dumps(dct))
        self.send(
            text_data=json.dumps({'activate': None}))


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


class ConfigWrapperConsumer(YAMLInitConsumer):
    pass

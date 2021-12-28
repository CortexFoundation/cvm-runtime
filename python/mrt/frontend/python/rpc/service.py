from concurrent import futures
import os
from os import path
from shutil import copyfile

import grpc

import rpc.service_pb2 as pb2
import rpc.service_pb2_grpc as pb2_grpc
from rpc.utils import get_streamer

# TODO(ryt): load balancer for maxinum_workers
maximum_workers = 4
#  local_addr = "127.0.0.1:5000"
# socket host difference
local_addr = "0.0.0.0:5000"
chunk_size = 1024 * 1024  # 1MB

def mrt_submit(
    src_sym_file, src_prm_file, dst_model_dir, host_addr=None):
    model_name = path.splitext(path.basename(src_sym_file))[0]
    model_name_2 = path.splitext(path.basename(src_prm_file))[0]
    assert model_name == model_name_2, "not compatible, " + \
        "src_sym_file: {}, src_prm_file: {}".format(
            src_sym_file, src_prm_file)
    if host_addr is None:
        dst_sym_file = path.join(dst_model_dir, model_name+".json")
        dst_prm_file = path.join(dst_model_dir, model_name+".params")
        copyfile(src_sym_file, dst_sym_file)
        copyfile(src_prm_file, dst_prm_file)
        yield "src files copied"
    else:
        def iterator_func(src_file, file_name):
            yield pb2.MRTClientReqStream(chunck=bytes(dst_model_dir, 'utf-8'))
            yield pb2.MRTClientReqStream(chunck=bytes(file_name, 'utf-8'))
            yield pb2.MRTClientReqStream(
                chunck=bytes(str(path.getsize(src_file)), 'utf-8'))
            with open(src_file, 'rb') as f:
                while True:
                    piece = f.read(chunk_size);
                    if len(piece) == 0:
                        return
                    yield pb2.MRTClientReqStream(chunck=piece)
        conn = grpc.insecure_channel(host_addr)
        client = pb2_grpc.MRTRpcSrvStub(channel=conn)
        response = client.submit(
            iterator_func(src_sym_file, model_name+".json"))
        next(response)
        for message in response:
            yield message.logging_str
        response = client.submit(
            iterator_func(src_prm_file, model_name+".params"))
        for message in response:
            yield message.logging_str

def mrt_execute(yaml_file_str, host_addr=None):
    if host_addr is None:
        my_streamer = get_streamer(yaml_file_str)
        for logging_str in my_streamer.start():
            yield logging_str
    else:
        conn = grpc.insecure_channel(host_addr)
        client = pb2_grpc.MRTRpcSrvStub(channel=conn)
        response = client.execute(
            pb2.MRTClientReq(content=yaml_file_str))
        for message in response:
            yield message.logging_str


class MRTRpcSrv(pb2_grpc.MRTRpcSrvServicer):
    def execute(self, request, context):
        yaml_file_str = request.content
        my_streamer = get_streamer(yaml_file_str)
        for message in my_streamer.start():
            if not context.is_active():
                raise RuntimeError("client connection lost")
            yield pb2.MRTServerResp(logging_str=message)
        #  if context.is_active():
            #  context.cancel()

    def submit(self, request_iterator, context):
        dst_model_dir = str(next(request_iterator).chunck, 'utf-8')
        os.makedirs(dst_model_dir, exist_ok=True)
        file_name = str(next(request_iterator).chunck, 'utf-8')
        size = eval(str(next(request_iterator).chunck, 'utf-8'))
        dst_file = path.join(dst_model_dir, file_name)
        with open(dst_file, 'wb') as f:
            cur_size = 0
            for piece in request_iterator:
                f.write(piece.chunck)
                cur_size += chunk_size
                cur_size = min(cur_size, size)
                message = "Current: {} Bytes / Total: {} Bytes, ".format(
                    cur_size, size) + \
                    "{} % Completed".format(round(cur_size/size*100.0, 2))
                yield pb2.MRTServerResp(logging_str=message)

def main():
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=maximum_workers))
    pb2_grpc.add_MRTRpcSrvServicer_to_server(
        MRTRpcSrv(), grpc_server)
    grpc_server.add_insecure_port(local_addr)
    grpc_server.start()
    print("server will start at {}".format(local_addr))
    grpc_server.wait_for_termination()

if __name__ == '__main__':
    main()

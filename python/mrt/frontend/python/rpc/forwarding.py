import os
import argparse

default_local_port = 5001
default_remote_port = 5000
default_remote_user = None
default_remote_host = None

parser = argparse.ArgumentParser()
parser.add_argument(
    "--local-port", type=int, default=default_local_port)
parser.add_argument(
    "--remote-port", type=int, default=default_remote_port)
parser.add_argument(
    "--remote-user", type=str, default=default_remote_user)
parser.add_argument(
    "--remote-host", type=str, default=default_remote_host)

def forward(
    local_port=default_local_port, remote_port=default_remote_port,
    remote_user=default_remote_user, remote_host=default_remote_host):
    if remote_user is None:
        raise RuntimeError("remote_user should be specified")
    if remote_host is None:
        raise RuntimeError("remote_host should be specified")
    cmd = "ssh -N -L {}:localhost:{} {}@{}".format(
        local_port, remote_port, remote_user, remote_host)
    os.system(cmd)

if __name__ == "__main__":
    args = parser.parse_args()
    forward(
        local_port=args.local_port, remote_port=args.remote_port,
        remote_user=args.remote_user, remote_host=args.remote_host)

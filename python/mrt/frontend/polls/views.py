from os import path

from django.http import HttpResponse

from . import views
from mrt.V3.utils import get_cfg_defaults
from mrt.V3.prepare import prepare

def views_prepare(request):
    yaml_file = path.expanduser("~/mrt_yaml_root/alexnet.yaml")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(yaml_file)
    cfg.freeze()
    cm_cfg = cfg.COMMON
    pass_cfg = cfg.PREPARE
    prepare(cm_cfg, pass_cfg)
    return HttpResponse("Hello, quantize")

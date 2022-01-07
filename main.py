from os import path
import sys

sys.path.insert(0, "./python")

from mrt.V3.utils import get_cfg_defaults, merge_cfg, override_cfg_args
from mrt.V3.execute import run
from mrt.V3.utils import DOC as utils_doc
from mrt.V3.prepare import DOC as prepare_doc
from mrt.V3.calibrate import DOC as calibrate_doc
from mrt.V3.quantize import DOC as quantize_doc
from mrt.V3.evaluate import DOC as evaluate_doc
from mrt.V3.mrt_compile import DOC as compile_doc

DOC = """
Usage:  python {0} --help
        python {0} [YAML_FILE_PATH] [OPTIONS]
""".format(sys.argv[0])

def complete_docs():
    return docs

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"]:
        docs = "\n".join([
            DOC, utils_doc, prepare_doc, calibrate_doc,
            quantize_doc, evaluate_doc, compile_doc])
        print(docs)
    else:
        assert len(sys.argv) >= 2 and len(sys.argv)%2 == 0, \
            "invalid length: {} of sys.argv: {}".format(
            len(sys.argv), sys.argv)
        yaml_file = sys.argv[1]
        cfg = get_cfg_defaults()
        cfg = merge_cfg(yaml_file)
        cfg = override_cfg_args(cfg, sys.argv[2:])
        run(cfg)

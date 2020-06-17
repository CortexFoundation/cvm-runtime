import os
import shutil

PROJECT_DIR = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))

BUILD_DIR = os.path.join(PROJECT_DIR, "build")
TESTS_DIR = os.path.join(BUILD_DIR, "tests")

if not os.path.exists(BUILD_DIR):
    os.mkdir(BUILD_DIR)

if not os.path.exists(TESTS_DIR):
    os.mkdir(TESTS_DIR)

CONFIG_PATH = os.path.join(PROJECT_DIR, "cmake/config.cmake")
LOCAL_CONFIG_PATH = os.path.join(PROJECT_DIR, "config.cmake")

if not os.path.exists(LOCAL_CONFIG_PATH):
    print ("Create local config path: ", LOCAL_CONFIG_PATH)
    shutil.copy(CONFIG_PATH, LOCAL_CONFIG_PATH)
elif os.path.getmtime(LOCAL_CONFIG_PATH) < os.path.getmtime(CONFIG_PATH):
    print ("Update local config path: ", LOCAL_CONFIG_PATH)
    shutil.copy(CONFIG_PATH, LOCAL_CONFIG_PATH)

PYTHONPATH = os.getenv("PYTHONPATH", "")
LD_LIBRARY_PATH = os.getenv("LD_LIBRARY_PATH", "")

try:
    import cvm
except ImportError as e:
    print (e)
    export_cmd = """

    Due to bash limitation, we cannot add python & link library
      environment via scripts, and then we supply the below commands to
      help to setup the project, copy and execute it in terminal or
      paste the commands into your bash profile and resource bashrc:

        export PYTHONPATH={}/python:${{PYTHONPATH}}
        export LD_LIBRARY_PATH={}/build:${{LD_LIBRARY_PATH}}

        """.format(PROJECT_DIR, PROJECT_DIR)

    print (export_cmd)
except Exception as e:
    pass

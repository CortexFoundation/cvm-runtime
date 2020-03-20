if [[ *"cvm-runtime/python"* != "${PYTHONPATH}" ]]; then
  export PYTHONPATH=`pwd`/python:${PYTHONPATH}
fi

echo "PYTHONPATH=${PYTHONPATH}"
echo

# compile the cython module
CURR_DIR=`pwd`
if [[ *${CURR_DIR}"/build"* != ${LD_LIBRARY_PATH} ]]; then
  export LD_LIBRARY_PATH=${CURR_DIR}/build:${LD_LIBRARY_PATH}
fi

echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo

python3 python/setup.py build_ext -i


CURR_DIR=`pwd`

if [[ *"${CURR_PATH}/python"* != "${PYTHONPATH}" ]]; then
  export PYTHONPATH=`pwd`/python:${PYTHONPATH}
fi

echo "PYTHONPATH=${PYTHONPATH}"
echo

if [[ *${CURR_DIR}"/build"* != ${LD_LIBRARY_PATH} ]]; then
  export LD_LIBRARY_PATH=${CURR_DIR}/build:${LD_LIBRARY_PATH}
fi

echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo

# compile the cython module
echo "Compile the cython setup"
cd python
python3 setup.py build_ext -i
cd ..

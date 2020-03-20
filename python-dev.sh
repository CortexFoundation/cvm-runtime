if [[ *"cvm-runtime/python"* != "${PYTHONPATH}" ]]; then
  export PYTHONPATH=`pwd`/python:${PYTHONPATH}
fi

echo ${PYTHONPATH}

# compile the cython module
cd python
CFLAGS="-I../include -I../"  LDFLAGS="-L../build/" python3 setup.py build_ext -i

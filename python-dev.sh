export PYTHONPATH=`pwd`/python:${PYTHONPATH}
echo ${PYTHONPATH}

# compile the cython module
cd python
CFLAGS="-I../include -I../"  LDFLAGS="-L../build/gpu/" python3 setup.py build_ext -i

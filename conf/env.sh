CURR_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))
ROOT_DIR=$(dirname ${CURR_DIR})
PROJECT_NAME=$(basename ${ROOT_DIR})

COM=""

if [[ "${PYTHONPATH}" != *"${PROJECT_NAME}/python"* ]]; then
  TPATH=${ROOT_DIR}/python
  [[ -z ${PYTHONPATH} ]] || TPATH=${TPATH}:${PYTHONPATH}
  export PYTHONPATH=${TPATH}
fi

if [[ "${LD_LIBRARY_PATH}" != *"${PROJECT_NAME}/build"* ]]; then
  TPATH=${ROOT_DIR}/build
  [[ -z ${LD_LIBRARY_PATH} ]] || TPATH=${TPATH}:${LD_LIBRARY_PATH}
  export LD_LIBRARY_PATH=${TPATH}
fi

# if [[ ${COM} != "" ]]; then
  # cat <<EOF

# Due to bash limitation, we cannot add python & link library
  # environment via scripts, and then we supply the below commands to
  # help to setup the project, copy and execute it in terminal please:

# \`
  # ${COM}
# \`

# EOF
# fi

# echo "Done."

# compile the cython module
# echo "Compile the cython setup"
# cd python
# python3 setup.py build_ext -i
# cd ..

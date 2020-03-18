#--------------------------------------------------------------------
#  Template custom cmake configuration for compiling
#
#  This file is used to override the build options in build.
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory. First copy the this
#  file so that any local changes will be ignored by git
#
#  $ mkdir build
#  $ cp cmake/config.cmake build
#
#  Next modify the according entries, and then compile by
#
#  $ cd build
#  $ cmake ..
#
#  Then buld in parallel with 8 threads
#
#  $ make -j8
#--------------------------------------------------------------------

#---------------------------------------------
# Backend runtimes.
#---------------------------------------------

set(PROFILE OFF)
set(DEBUG OFF)
set(USE_FORMAL OFF)
# Whether enable CUDA during compile,
#
# Possible values:
# - ON: enable CUDA with cmake's auto search
# - OFF: disbale CUDA
# - /path/to/cuda: use specific path to cuda toolkit
set(USE_CUDA OFF)

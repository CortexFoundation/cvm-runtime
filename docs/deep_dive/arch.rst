
.. _design-and-arch:

***********************
Design and Architecture
***********************

This document is intended for developers who want to understand the architecture of cvm-runtime and/or actively develop on the project. This page is organized as follows:

- The :ref:`system-design` introduces all the important details
  of system design. To get started, please read this section first.

- The :ref:`example-infer-flow` gives an overview of the steps
  that CVM takes to load a hard-disk model into CPU/GPU memory
  for running the deterministic result.

- The :ref:`operator-math-formalization` section describes the ideal
  process logic for specific input data with strict mathematical
  description, guiding the source code.

.. toctree::
  :maxdepth: 2
  
  system
  infer_flow
  math_formalization


.. _operator-test:

*************
Operator Test
*************

.. contents::

Test Method
-----------

First generate test data with test_ops.py, then verify the 
generated data with test_op.cc, execute the following code 
to test(test_op.cc is compiled with Makefile):

.. code-block::

    python tests/test_ops.py
    ./test_op 0/1/2 (0 is cpu, 1 is formal, and 2 is gpu)

Test Results
------------

The following are the test results of the operator on different devices:

+-------------------+-------------------------+
|     operator      |          results        |
+===================+=========================+
| broadcast_sub     | cpu & formal & gpu pass |
+-------------------+-------------------------+
| broadcast_add     | cpu & formal & gpu pass |
+-------------------+-------------------------+
| broadcast_mul     | cpu & formal & gpu pass |
+-------------------+-------------------------+
| broadcast_max     | cpu & formal & gpu pass |
+-------------------+-------------------------+
| broadcast_div     | cpu & formal & gpu pass |
+-------------------+-------------------------+
| broadcast_greater | cpu & formal & gpu pass |
+-------------------+-------------------------+
| max_pool2d        | cpu & formal & gpu pass |
+-------------------+-------------------------+
| dense             | cpu & formal & gpu pass |
+-------------------+-------------------------+
| sum               | cpu & formal & gpu pass |
+-------------------+-------------------------+
| max               | cpu & formal & gpu pass |
+-------------------+-------------------------+
| slice_like        | cpu & formal & gpu pass |
+-------------------+-------------------------+
| tile              | cpu & formal & gpu pass |
+-------------------+-------------------------+
| repeat            | cpu & formal & gpu pass |
+-------------------+-------------------------+
| concatenate       | cpu & formal & gpu pass |
+-------------------+-------------------------+
| transpose         | cpu & formal & gpu pass |
+-------------------+-------------------------+
| take              | cpu & formal & gpu pass |
+-------------------+-------------------------+
| get_valid_counts  | cpu & formal & gpu pass |
+-------------------+-------------------------+
| strided_slice     | cpu & formal & gpu pass |
+-------------------+-------------------------+

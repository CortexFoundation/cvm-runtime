"""
Get Started with NNVM
=====================
**Author**: `Tianqi Chen <https://tqchen.github.io/>`_

This article is an introductory tutorial to workflow in NNVM.
"""
import nnvm.compiler
import nnvm.symbol as sym

######################################################################
# Declare Computation
# -------------------
# We start by describing our need using computational graph.
# Most deep learning frameworks use computation graph to describe
# their computation. In this example, we directly use
# NNVM's API to construct the computational graph.
#
# .. note::
#
#   In a typical deep learning compilation workflow,
#   we can get the models from :any:`nnvm.frontend`
#
# The following code snippet describes :math:`z = x + \sqrt{y}`
# and creates a nnvm graph from the description.
# We can print out the graph ir to check the graph content.

x = sym.Variable("data")
y = sym.clip(x, a_min=0, a_max=127)
compute_graph = nnvm.graph.create(y)
print("-------compute graph-------")
print(compute_graph.ir())

######################################################################
# Compile
# -------
# We can call :any:`nnvm.compiler.build` to compile the graph.
# The build function takes a shape parameter which specifies the
# input shape requirement. Here we only need to pass in shape of ``x``
# and the other one will be inferred automatically by NNVM.
#
# The function returns three values. ``deploy_graph`` contains
# the final compiled graph structure. ``lib`` is a :any:`tvm.module.Module`
# that contains compiled CUDA functions. We do not need the ``params``
# in this case.
shape = (1, 28)
deploy_graph, lib, params = nnvm.compiler.build(
    compute_graph, target="cuda", target_host="llvm --system-lib", shape={"data": shape}, dtype="int32")
print (deploy_graph.json())

from typing import List, Callable
from typing import TypeVar

NodeT = TypeVar("node")
GraphT = List[NodeT]

RefsFuncT = Callable[[NodeT], GraphT]

CycleNodeT = GraphT
CyclingFuncT = Callable[[GraphT], None]

VisitFuncT = Callable[[NodeT, int, int], None]

def _default_cycling_trigger(graph : CycleNodeT):
    raise RuntimeError(
            "graph has cycle with path: {}->{}".format(
                "->".join([str(n) for n in graph]),
                str(graph[0])))

def _default_visit_func(
        node : NodeT,
        ref_size : int, # all child size
        # current visit sequence index, range [0, child_size]
        index : int):
    pass

def dfs_visit(
        graph : GraphT,
        refs_generator: RefsFuncT,
        visit_func : VisitFuncT = _default_visit_func,
        cycling_trigger : CyclingFuncT = _default_cycling_trigger):
    """ Abstract Graph DFS Visit Algorithm

        The solution is deep first sequence, with quick leaf
            cut of set visited attribute.

        Parameters
        ==========
        graph: node array, and node should support stringlify.
        refs_generator: function with node as parameter,
            to get the reference nodes for input node.
    """
    visited_nodes : GraphT = []
    for node in list(graph):
        _dfs_impl(node,
                  refs_generator,
                  visit_func,
                  cycling_trigger,
                  visited_nodes)

def _dfs_impl(
        node : NodeT,
        refs_generator : RefsFuncT,
        visit_func : VisitFuncT,
        cycling_trigger : CyclingFuncT,
        visited_nodes : GraphT,
        dfs_path : GraphT = []):
    if node in visited_nodes:
        return

    dfs_path.append(node)

    ref_nodes = refs_generator(node)
    ref_size = len(ref_nodes)

    for idx, ref_node in enumerate(list(ref_nodes)):
        # dfs visit function, interface format
        visit_func(node, ref_size, idx)

        # cycling to skip, process after the dfs visit
        #   for current node
        if ref_node in dfs_path:
            cycling_trigger(dfs_path)
            continue

        _dfs_impl(ref_node,
                  refs_generator,
                  visit_func,
                  cycling_trigger,
                  visited_nodes,
                  dfs_path)

    # last visit function, the ref_size ranged in [0, N]
    visit_func(node, ref_size, ref_size)

    dfs_path.pop(-1)
    visited_nodes.append(node)

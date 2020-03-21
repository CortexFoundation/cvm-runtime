/*!
 *  Copyright (c) 2019 by Contributors
 * \file infer_precision.cc
 * \brief Inference the precisions given existing information.
 */
#include <cvm/pass.h>
#include <cvm/op_attr_types.h>
#include <cvm/graph_attr_types.h>
#include <unordered_map>
#include <iostream>

namespace cvm {
namespace pass {
namespace {

    /*
inline bool InferPrecisionDefault(const NodeAttrs& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
    std::cout << attrs.op->name << std::endl;
    return true;
}
*/
/*
typedef bool InferFunc(std::vector<int>*, std::vector<int>*);

std::unordered_map<std::string,InferFunc> InferFuncMap({
        {
            "clip",
            [bool](std::vector<int>* iattr, std::vector<int>* oattr){
                return true;
            }
        },
        });
*/
inline bool InferPrecisionForward(const NodeAttrs& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
    // std::cout << attrs.op->name << ' ' << iattr->size() << ' ' << oattr->size() << std::endl;

    return true;
}

inline bool InferPrecisionBackward(const NodeAttrs& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
    std::cout << attrs.op->name << std::endl;
    return true;
}

inline bool is_none(const int t){ return t == -1; }

Graph InferPrecision(Graph &&ret) {
  using AttrVector = std::vector<int>;
  const IndexedGraph& idx = ret.indexed_graph();
  static auto& finfer_prec =
      Op::GetAttr<FInferNodeEntryAttr<int> >("FInferPrecision");
  static auto& is_backward =
      Op::GetAttr<TIsBackward>("TIsBackward");
  // gradient function, used to get node correspondence.
  //static auto& fgrad =
  //   Op::GetAttr<FGradient>("FGradient");
  // reshape shape vector
  AttrVector rshape;
  if (ret.attrs.count("precision") != 0) {
    rshape = ret.MoveCopyAttr<AttrVector>("precision");
  } else {
    rshape.resize(idx.num_node_entries(), -1);
  }

  // get the shape hints
  std::string shape_hints_key = std::string("precision") + "_hints";
  if (ret.attrs.count(shape_hints_key)) {
    NodeEntryMap<int> shape_hints =
      ret.GetAttr<NodeEntryMap<int>>(shape_hints_key);
    for (const auto& kv : shape_hints) {
      NodeEntry e = kv.first;
      if (idx.exist(e.node.get())) {
        rshape[idx.entry_id(kv.first)] = kv.second;
      }
    }
  }

  std::string shape_attr_key;
  if (ret.attrs.count("precision_attr_key") != 0) {
    shape_attr_key = ret.GetAttr<std::string>("precision_attr_key");
    // erase the provided arguments
    ret.attrs.erase("precision_attr_key");
  } else {
    shape_attr_key = "precision";
  }
  // Temp space for shape inference.
  std::vector<int> ishape, oshape;

  // inference step function for nid
  auto infer_step = [&](uint32_t nid, bool last_iter) {
    const auto& inode = idx[nid];
    const uint32_t num_inputs = inode.inputs.size();
    const uint32_t num_outputs = inode.source->num_outputs();
    if (inode.source->is_variable()) {
      // Variable node. No operator. Only one output entry.
      CHECK(inode.source->op() == nullptr);
      CHECK_EQ(num_outputs, 1U);
      const uint32_t out_ent_id = idx.entry_id(nid, 0);
//      std::cout << "Variable at " << nid << " " <<  inode.source->attrs.name << std::endl;
      if (shape_attr_key.length() != 0 && is_none(rshape[out_ent_id])) {
        auto it = inode.source->attrs.dict.find(shape_attr_key);
        if (it != inode.source->attrs.dict.end()) {
          std::istringstream is(it->second);
          CHECK(is >> rshape[out_ent_id]) << "Invalid attribute";
        }
      }
    } else if (is_backward.get(inode.source->op(), false) && inode.control_deps.size()) {
      //CHECK_GE(inode.control_deps.size(), 1U)
      //  << "BackwardOp need to have control_deps to its forward op";
      //const IndexedGraph::Node& fnode = idx[inode.control_deps[0]];
      //NodePtr fwd_ptr = inode.source->control_deps[0];
      //CHECK(fwd_ptr->op() != nullptr) << "Forward op cannot be a variable";
      //// use gradient function to find out the correspondence.
      //std::vector<NodeEntry> ograd(fwd_ptr->num_outputs());
      //for (size_t i = 0; i < ograd.size(); ++i) {
      //  ograd[i].index = static_cast<uint32_t>(i);
      //}
      //// input gradient list
      //auto igrad = fgrad[fwd_ptr->op()](fwd_ptr, ograd);
      //const Node* igrad_node = nullptr;
      //// Input gradient assignement
      //for (size_t i = 0; i < igrad.size(); ++i) {
      //  if (igrad[i].node->op() == inode.source->op()) {
      //    uint32_t eid = idx.entry_id(nid, igrad[i].index);
      //    if (is_none(rshape[eid])) {
      //      rshape[eid] = rshape[idx.entry_id(fnode.inputs[i])];
      //    } else if (!is_none(rshape[idx.entry_id(fnode.inputs[i])])) {
      //      CHECK_EQ(rshape[eid], rshape[idx.entry_id(fnode.inputs[i])])
      //          << "Backward shape inconsistent with the forward shape";
      //    }
      //    if (igrad_node == nullptr) {
      //      igrad_node = igrad[i].node.get();
      //    } else {
      //      CHECK(igrad_node == igrad[i].node.get());
      //    }
      //  }
      //}
      //// out grad entries
      //CHECK(igrad_node != nullptr)
      //  << "Cannot find matching backward op for " << inode.source->attrs.name;
      //for (size_t i = 0; i < igrad_node->inputs.size(); ++i) {
      //  const NodeEntry& e = igrad_node->inputs[i];
      //  if (e.node == nullptr) {
      //    uint32_t eid = idx.entry_id(inode.inputs[i]);
      //    if (is_none(rshape[eid])) {
      //      rshape[eid] = rshape[idx.entry_id(inode.control_deps[0], e.index)];
      //    }
      //  }
      //}
    } else {
      bool forward_known = true;
      // Forward operator inference.
      ishape.resize(num_inputs, -1);
      for (uint32_t i = 0; i < ishape.size(); ++i) {
        ishape[i] = rshape[idx.entry_id(inode.inputs[i])];
        if (is_none(ishape[i])) forward_known = false;
      }
      oshape.resize(num_outputs, -1);
      for (uint32_t i = 0; i < oshape.size(); ++i) {
        oshape[i] = rshape[idx.entry_id(nid, i)];
        if (is_none(oshape[i])) forward_known = false;
      }
      auto finfer = finfer_prec.get(inode.source->op(), InferPrecisionForward);
      if (!forward_known) {
        if (finfer != nullptr) {
          // Call inference function of the operator.
          try {
            forward_known = finfer(inode.source->attrs, &ishape, &oshape);
          } catch (const std::exception& e) {
            throw utils::Error("Error in operator " + inode.source->attrs.name + ": " + e.what());
          }
        } else {
          CHECK(!last_iter)
              << "Attribute " << "FInferPrecision"
              << " is not registered by op " << inode.source->op()->name
              << " we are not able to complete the inference because of this";
        }
      }
      // Save to the result map.
      for (uint32_t i = 0; i < num_inputs; ++i) {
        rshape[idx.entry_id(inode.inputs[i])] = ishape[i];
      }
      for (uint32_t i = 0; i < num_outputs; ++i) {
        rshape[idx.entry_id(nid, i)] = oshape[i];
      }
    }
  };

  size_t last_num_unknown;
  size_t num_unknown = rshape.size();
  int i = 0;
  do {
    if (i % 2 == 0) {
      for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
        infer_step(nid, false);
      }
    } else {
      // backward inference
      for (uint32_t i = idx.num_nodes(); i != 0; --i) {
        infer_step(i - 1, false);
      }
    }
    last_num_unknown = num_unknown;
    num_unknown = 0;
    for (size_t j = 0; j < idx.num_node_entries(); ++j) {
      if (is_none(rshape[j])) {
        ++num_unknown;
      }
    }
    ++i;
  } while (num_unknown > 0 && last_num_unknown > num_unknown);
  // set the precisions
  ret.attrs["precision"] = std::make_shared<any>(std::move(rshape));
  // number of nodes who knows the precision.
  ret.attrs["precision_num_unknown_nodes"] = std::make_shared<any>(num_unknown);
  return std::move(ret);
}

CVM_REGISTER_PASS(InferPrecision)
.describe("Infer the precesion of each node entries.")
.set_body([](Graph ret) {
    return InferPrecision(std::move(ret));
  })
.set_change_graph(false)
.provide_graph_attr("precision");

CVMUTIL_JSON_ENABLE_ANY(size_t, size_t);

}  // namespace
}  // namespace pass
}  // namespace cvm

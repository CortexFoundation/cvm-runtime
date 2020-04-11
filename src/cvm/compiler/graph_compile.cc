/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph_compile.cc
 * \brief Compile a graph. It lowers the graph nodes into low level IR.
 */
#include <sstream>
#include <utils/parameter.h>
//#include <cvm/compiler/packed_func_ext.h>
#include <cvm/graph.h>
#include <cvm/graph_attr_types.h>
#include <cvm/node.h>
#include <cvm/op_attr_types.h>
#include <cvm/pass.h>
#include <cvm/pass_functions.h>
#include <cvm/tuple.h>
//#include <tvm/lowered_func.h>
#include <cvm/runtime/packed_func.h>
#include "../graph_runtime.h"

#include "graph_fuse.h"
#include "pattern_util.h"


namespace cvm {
namespace compiler {

using namespace cvm;

// Decorate the result of PlanMemory
// This function does two things:
// - Give separate memory to each variable.
// - Tie the memory of output/lhs in assign node properly
//   so the execution of assign can have side effect.
cvm::Graph DecorateMemoryPlan(
    cvm::Graph g,
    const std::vector<int>& assign_flag) {
  const IndexedGraph& idx = g.indexed_graph();
  StorageVector storage_vec = g.MoveCopyAttr<StorageVector>("storage_id");
  g.attrs.erase("storage_allocated_bytes");
  g.attrs.erase("storage_inplace_index");
  size_t num_not_allocated = g.MoveCopyAttr<size_t>(
      "storage_num_not_allocated");
  CHECK_EQ(num_not_allocated, 0U)
      << "Can only build inference graph with all statically allocated memory";

  // Reassign variable id so that they are different.
  int max_id = 0;
  for (size_t i = 0; i < storage_vec.size(); ++i) {
    max_id = std::max(storage_vec[i] + 1, max_id);
  }
  for (uint32_t nid : idx.input_nodes()) {
    storage_vec[idx.entry_id(nid, 0)] = max_id++;
  }
  // Tie up the assign node storage properly.
  for (uint32_t nid = 0 ; nid < idx.num_nodes(); ++nid) {
    if (assign_flag[nid] == 0) continue;
    const auto& inode = idx[nid];
    int var_storage_id = storage_vec[idx.entry_id(inode.inputs[0])];
    storage_vec[idx.entry_id(nid, 0)] = var_storage_id;

    if (assign_flag[nid] == 2) {
      storage_vec[idx.entry_id(inode.inputs[1])] = var_storage_id;
    }
  }
  g.attrs["storage_id"] = std::make_shared<any>(std::move(storage_vec));
  return g;
}

// Get unique name
std::string GetUniqeName(
    std::unordered_map<std::string, int> &name_map, 
    std::string name) {
  auto it = name_map.find(name);
  std::ostringstream os;
  if (it == name_map.end()) {
    name_map[name] = 0;
    os << name << "_0";
  } else {
    ++(it->second);
    os << name << "_" << it->second;
  }
  name = os.str();
  return name;
}

cvm::Graph GraphCompile(const cvm::Graph& g) {
  printf("start graph compile\n");
  // Get attributes from the graph.
  const ShapeVector& shape_vec = g.GetAttr<ShapeVector>("shape");
  const DTypeVector& dtype_vec = g.GetAttr<DTypeVector>("dtype");
  const DTypeVector& precision_vec = g.GetAttr<DTypeVector>("precision");
  //const GroupVec& group_vec = g.GetAttr<GroupVec>("group_root");
  //const PatternVec& pattern_vec = g.GetAttr<PatternVec>("pattern");
  std::unordered_map<std::string, int> name_map;

  //CHECK(g.HasAttr("fused_entry")) << "Fusion hasn't been applied yet.";
  //FuseEntryVec fuse_entries = g.GetAttr<FuseEntryVec>("fused_entry");

  // collect op attributes
  const IndexedGraph& idx = g.indexed_graph();
  std::vector<int> prec_vec(idx.num_node_entries(), -1);
  std::vector<std::string> op_attrs(idx.num_node_entries(), "{}");
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    printf("\"name\":\"%s\"\n", inode.source->attrs.name.c_str());
    const auto& attrs_dict = inode.source->attrs.dict;
    //auto search = attrs_dict.find("precision");
    //int precision = -1;
    //if (search != attrs_dict.end()) {
    //  CHECK_EQ(inode.source->num_outputs(), 1U)
    //    << "variable precision must be 1 outputs";
    //  precision = std::stoi(search->second);
    //}
    //for (uint32_t i = 0; i < inode.source->num_outputs(); ++i) {
    //  uint32_t eid = idx.entry_id(nid, i);
    //  prec_vec[eid] = precision;
    //}
    std::vector<std::string> attr_vec;
    for (auto& item: inode.source->attrs.dict) {
        std::stringstream tss;
        tss << "\"" << item.first << "\": " << "\"" << item.second << "\"";
        attr_vec.push_back(tss.str());
    }
    std::stringstream ss;
    ss << "{";
    for (size_t i = 0; i < attr_vec.size(); i++) {
        if (i != 0) ss << ", ";
        ss << attr_vec[i];
    }
    ss << "}";
    std::string attrs = ss.str();
    op_attrs[nid] = attrs;
    printf("%s\n", attrs.c_str());
  }

  //const cvm::Op* cvm_op = cvm::Op::Get("cvm_op");
  std::unordered_map<uint32_t, cvm::NodePtr> old_new;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) {
      // Only copy name since that is sufficient.
      cvm::NodePtr np = cvm::Node::Create();
      np->attrs.name = inode.source->attrs.name;
      old_new[nid] = np;
      continue;
    }
    //int root_id = group_vec[nid];
    //if (static_cast<int>(nid) != root_id) continue;

    //// Handle normal op
    //FuseEntry& fe = fuse_entries[root_id];
    //const IndexedGraph& subidx = fe.subgraph.indexed_graph();
    cvm::NodePtr np = cvm::Node::Create();
    //np->attrs.op = cvm_op;
    auto& op_name = inode.source->attrs.op->name;
    np->attrs.op = cvm::Op::Get(op_name);
    np->attrs.name = GetUniqeName(name_map, op_name);
    runtime::CVMOpParam param;
    param.func_name = op_name;
    //param.num_inputs = static_cast<uint32_t>(fe.imap.size());
    //param.num_outputs = static_cast<uint32_t>(fe.subgraph.outputs.size());
    //param.flatten_data = fe.flatten_data;
    param.num_inputs = np->attrs.op->num_inputs;
    param.num_outputs = np->attrs.op->num_outputs;
    param.flatten_data = 0;
    param.UpdateDict(&(np->attrs.dict));
    np->attrs.parsed = std::move(param);

    //for (uint32_t sub_input_id : subidx.input_nodes()) {
    //  // Need to make sure subgraph input order is consistent to the order of
    //  // the graph input.
    //  auto rit = fe.reverse_imap.find(subidx[sub_input_id].source);
    //  CHECK(rit != fe.reverse_imap.end());
    //  const IndexedGraph::NodeEntry& e = rit->second;
    //  auto it = old_new.find(e.node_id);
    //  CHECK(it != old_new.end())
    //      << "cannot find node_id=" << e.node_id;
    //  np->inputs.emplace_back(
    //      cvm::NodeEntry{it->second, e.index, e.version});
    //}
    for (const uint32_t node_id : inode.control_deps) {
      auto it = old_new.find(node_id);
      CHECK(it != old_new.end());
      np->control_deps.emplace_back(it->second);
    }
    old_new[nid] = np;
  }

  cvm::Graph ret;
  for (const auto& e : idx.outputs()) {
    //auto it = old_new.find(group_vec[e.node_id]);
    auto it = old_new.find(e.node_id);
    CHECK(it != old_new.end())
        << "cannot find node_id=" << e.node_id;
    ret.outputs.emplace_back(
        cvm::NodeEntry{it->second, e.index, e.version});
  }

  // Reference counter of each op node.
  // For now, always store result when an op is referred more than once.
  std::vector<uint32_t> ref_count = GetNodeRefCounts(idx);
  for (const auto& e : idx.outputs()) {
    // This line will realize all the outputs.
    ref_count[e.node_id] += 1;
  }

  const IndexedGraph& new_idx = ret.indexed_graph();

  ShapeVector new_shape_vec = ShapeVector(new_idx.num_node_entries(), TShape());
  DTypeVector new_dtype_vec = DTypeVector(new_idx.num_node_entries());
  std::vector<int> new_prec_vec(new_idx.num_node_entries(), -1);
  std::vector<std::string> new_dltype_vec(new_idx.num_node_entries());
  std::vector<std::string>  new_op_attrs(new_idx.num_nodes());
  for (const auto& kv : old_new) {
    uint32_t nid = kv.first;
    const auto& inode = idx[nid];
    uint32_t new_nid = new_idx.node_id(kv.second.get());
    //new_op_attrs[new_nid] = op_attrs[nid];
    //for (uint32_t i = 0; i < inode.source->num_outputs(); ++i) {
    //  uint32_t new_eid = new_idx.entry_id(new_idx.node_id(kv.second.get()), i);
    //  uint32_t old_eid = idx.entry_id(nid, i);
    //  new_shape_vec[new_eid] = shape_vec[old_eid];
    //  new_dtype_vec[new_eid] = dtype_vec[old_eid];
    //  new_prec_vec[new_eid] = prec_vec[old_eid];
    //  new_dltype_vec[new_eid] = "int32";
    //}
  }

  ret.attrs["shape"] = std::make_shared<any>(std::move(new_shape_vec));
  ret.attrs["dtype"] = std::make_shared<any>(std::move(new_dtype_vec));

  ret = cvm::ApplyPass(ret, "PlanMemory");
  //ret = DecorateMemoryPlan(ret, assign_flag);

  CHECK_EQ(new_idx.num_nodes(), new_op_attrs.size())
    << "OpAttrs is not consistant with nodes " << new_idx.num_nodes()
    << " vs. " << new_op_attrs.size();
  ret.attrs.erase("dtype");
  ret.attrs["precision"] = std::make_shared<any>(std::move(new_prec_vec));
  ret.attrs["dltype"] = std::make_shared<any>(std::move(new_dltype_vec));
  ret.attrs["op_attrs"] = std::make_shared<any>(std::move(new_op_attrs));
  printf("end graph compile\n");
  return ret;
}

CVM_REGISTER_PASS(GraphCompile)
    .set_body(GraphCompile)
    .depend_graph_attr("shape")
    .depend_graph_attr("dtype");
//    .depend_graph_attr("fused_entry")
//    .depend_graph_attr("group_root")
//    .depend_graph_attr("pattern");


}  // namespace compiler
}  // namespace cvm

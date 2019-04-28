/*!
 *  Copyright (c) 2017 by Contributors
 * \file rpc_server_env.cc
 * \brief Server environment of the RPC.
 */
#include <cvm/runtime/registry.h>
#include "../file_util.h"

namespace cvm {
namespace runtime {

std::string RPCGetPath(const std::string& name) {
  static const PackedFunc* f =
      runtime::Registry::Get("cvm.rpc.server.workpath");
  CHECK(f != nullptr) << "require cvm.rpc.server.workpath";
  return (*f)(name);
}

CVM_REGISTER_GLOBAL("cvm.rpc.server.upload").
set_body([](CVMArgs args, CVMRetValue *rv) {
    std::string file_name = RPCGetPath(args[0]);
    std::string data = args[1];
    SaveBinaryToFile(file_name, data);
  });

CVM_REGISTER_GLOBAL("cvm.rpc.server.download")
.set_body([](CVMArgs args, CVMRetValue *rv) {
    std::string file_name = RPCGetPath(args[0]);
    std::string data;
    LoadBinaryFromFile(file_name, &data);
    CVMByteArray arr;
    arr.data = data.c_str();
    arr.size = data.length();
    LOG(INFO) << "Download " << file_name << "... nbytes=" << arr.size;
    *rv = arr;
  });

CVM_REGISTER_GLOBAL("cvm.rpc.server.remove")
.set_body([](CVMArgs args, CVMRetValue *rv) {
    std::string file_name = RPCGetPath(args[0]);
    RemoveFile(file_name);
  });

}  // namespace runtime
}  // namespace cvm

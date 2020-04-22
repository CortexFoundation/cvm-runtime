#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <vector>
#include <string>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

inline std::vector<unsigned char> read_binary_file(const std::string& xclbin_file_name) {
    //std::cout << "INFO: Reading " << xclbin_file_name << std::endl;

    if (access(xclbin_file_name.c_str(), R_OK) != 0) {
        printf("ERROR: %s xclbin not available please build\n",
               xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    //Loading XCL Bin into char buffer
    //std::cout << "Loading: '" << xclbin_file_name.c_str() << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    auto nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    std::vector<unsigned char> buf;
    buf.resize(nb);
    bin_file.read(reinterpret_cast<char *>(buf.data()), nb);
    return buf;
}
#endif

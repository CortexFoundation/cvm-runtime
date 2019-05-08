#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <stdint.h>
#include <string.h>
using namespace std;


int main(){
    std::ifstream params_in("/tmp/inception_v3/params", std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    //read header, reserved
    uint64_t header, reserved;
    const char *p = params_data.c_str();
    memcpy(&header, p, sizeof(uint64_t));
    p += sizeof(uint64_t);
    memcpy(&reserved, p, sizeof(uint64_t));
    p += sizeof(uint64_t);
    cout << "header= " << header << " ,reserved=" << reserved << endl;

    //read names
    uint64_t sz;
    memcpy(&sz, p, sizeof(uint64_t));
    p += sizeof(uint64_t);
    size_t size = static_cast<size_t>(sz);
    cout << "name size = " << size << endl;
    vector<string> names;
    names.resize(size);
    for(int i = 0; i < size; i++){
        uint64_t tsz;
        memcpy(&tsz, p, sizeof(uint64_t));
        p += sizeof(uint64_t);
        string str;
        str.resize(tsz*sizeof(char));
        memcpy(&str[0], p, tsz);
        names[i] = str;
        p += tsz;
    }
    uint64_t vec_name_size;
    memcpy(&vec_name_size, p, sizeof(uint64_t));
    p += sizeof(uint64_t);
    cout << "vec_name_size=" << vec_name_size << " ,names: " << endl;
    for(int i = 0; i < vec_name_size; i++){
        cout << names[i] << endl;
    }

    //read data
    cout << "read data:\n";
    for(int i = 0; i < vec_name_size; i++){
        uint64_t header, reversed;
        memcpy(&header, p, sizeof(uint64_t));
        p += sizeof(uint64_t);
        memcpy(&reversed, p, sizeof(uint64_t));
        p += sizeof(uint64_t);
        cout << "header=" << header << ", reversed=" << reversed << endl;

        int device_type, device_id, ndim;
        memcpy(&device_type, p, sizeof(int));
        p += sizeof(int);
        memcpy(&device_id, p, sizeof(int));
        p += sizeof(int);
        memcpy(&ndim, p, sizeof(int));
        p += sizeof(int);
        cout << "device_type=" << device_type << ", device_id=" << device_id << ", ndim=" << ndim << endl;

        int8_t code, bits;
        memcpy(&code, p , sizeof(int8_t));
        p += sizeof(int8_t);
        memcpy(&bits, p, sizeof(int8_t));
        p += sizeof(int8_t);
        cout << "code=" << (int)code << ", bits=" << (int)bits << endl;

        int16_t lanes;
        memcpy(&lanes, p, sizeof(int16_t));
        p += sizeof(int16_t);
        cout << "lanes=" << lanes << endl;

        vector<int64_t> shape;
        shape.resize(ndim * sizeof(int64_t));
        memcpy(&shape[0], p , sizeof(int64_t) * ndim);
        p += sizeof(int64_t) * ndim;
        cout << "shape:\n";
        for(int i = 0; i < ndim; i++)
           cout <<  shape[i] << " ";
        cout << endl;

        int64_t data_byte_size;
        memcpy(&data_byte_size, p, sizeof(int64_t));
        p += sizeof(int64_t);
        cout << "data_byte_size=" << data_byte_size << endl;

        if(bits == 8){
            vector<int8_t>data;
            int64_t num_elems = 1;
            for(int i = 0; i < ndim; i++){
                num_elems *= shape[i];
            }
            cout << "num_elemems * sizeof(int) =" << num_elems * sizeof(int8_t) << endl;
            data.resize(num_elems);
            memcpy(&data[0], p , data_byte_size);
            p += data_byte_size;
        }else{
            vector<int>data;
            int64_t num_elems = 1;
            for(int i = 0; i < ndim; i++){
                num_elems *= shape[i];
            }
            cout << "num_elemems * sizeof(int) =" << num_elems * sizeof(int) << endl;
            data.resize(num_elems);
            memcpy(&data[0], p , data_byte_size);
            p += data_byte_size;

        }
/*        for(int j = 0 ; j < num_elems; j++){
            cout << data[j] << " ";
        }
        cout << endl;
        */
    }
    return 0;
}

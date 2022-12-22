#include "test_op.cc"

// NDArray conv2d(NDArray const& data, NDArray const& weight, NDArray* bias = nullptr, vector<int> padding, vector<int> strides) 

void handle(vector<vector<uint64_t>> tshape, vector<vector<int32_t>> tdata, vector<DLTensor> args, CVMOpParam params, string attr_str, DLTensor* extra_space)
{
    DLTensor* cpu_tensor;
    for(int i = 0; i < args.size(); i++)
    {
        DLTensor* dl;
        CVMArrayAlloc((int64_t*)tshape[i].data(), tshape[i].size(), dtype_code, dtype_bits, dtype_lanes, ctx, device_id, &dl);
        args[i] = *dl;
        if(i < params.num_inputs)
        {
            CVMArrayAlloc((int64_t*)tshape[i].data(), tshape[i].size(), dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
            memcpy(cpu_tensor->data, tdata[i].data(), sizeof(int32_t)*tdata[i].size());
            CVMArrayCopyFromTo(cpu_tensor, dl, nullptr);
            CVMArrayFree(cpu_tensor);
        }
    }
    NodeAttrs attr;
    LoadOp(params.func_name, attr);
    LoadOpAttr(attr_str, attr);
    auto op = get_func(params, &attr, args, params.num_inputs, extra_space);
    op();
    for (uint out_no = 0; out_no < params.num_outputs; out_no++)
    {
        vector<int32_t> cpu_output_tensor(tdata[params.num_inputs].size());
        {
            int i = params.num_inputs;
            CVMArrayAlloc((int64_t*)tshape[i].data(), tshape[i].size(), dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
            CVMArrayCopyFromTo(&args[i], cpu_tensor, nullptr);
            memcpy(cpu_output_tensor.data(), cpu_tensor->data, sizeof(int32_t)*tdata[i].size());
            printf("call CVMArrayFree by manual.....\n");
            CVMArrayFree(cpu_tensor);
        }
        int ret = memcmp(cpu_output_tensor.data(),
                        tdata[params.num_inputs].data(),
                        sizeof(int32_t) * tdata[params.num_inputs].size());
        printf("match %d | %d\n", ret == 0, ret);
        if (ret != 0)
        {
            for(uint i = 0; i < params.num_inputs; i++)
            {
                printf("input%d:\n", i);
                print(tdata[i]);
            }
            printf("correct out:");
            print(tdata[params.num_inputs+out_no]);
            printf("     my out:");
            print(cpu_output_tensor);
        }
        assert(ret == 0);
        CVMArrayFree(extra_space);
    }
    cout << params.func_name << " test finished" << endl;
}

void conv2d(NDArray const& data, NDArray const& weight, string attr_str, NDArray* bias = nullptr)
{
    vector<vector<int>> tshape;
    vector<vector<int>> tdata;
    int32_t *ddata = static_cast<int32_t*>(data->data);
    vector<int> dshape = vector<int>(data->shape, data->shape + data->ndim);
    tshape.push_back(dshape);
    vector<int> d_data;
    int dsize = dshape[0]*dshape[1]*dshape[2]*dshape[3];
    for (int i = 0; i < dsize; i++)
    {
        d_data.push_back(*ddata + i);
    }
    tdata.push_back(d_data);
    vector<int> wshape = vector<int>(weight->shape, weight->shape + weight->ndim);
    tshape.push_back(wshape);
    int32_t *wdata = static_cast<int32_t *>(weight->data);
    vector<int> w_data;
    int wsize = wshape[0]*wshape[1]*wshape[2]*wshape[3];
    for (int i = 0; i < wsize; i++)
    {
        w_data.push_back(*wdata + i);
    }
    tdata.push_back(w_data);
    if (bias != nullptr)
    {
        DLTensor *bptr = bias->operator->();
        vector<int> bshape = vector<int>(bptr->shape, bptr->shape + bptr->ndim);
        tshape.push_back(bshape);
        int32_t *bdata = static_cast<int32_t *>(bptr->data);
        vector<int> b_data;
        int bsize = bshape[0];
        for (int i = 0; i < bsize; i++)
        {
            b_data.push_back(*bdata + i);
        }
        tdata.push_back(b_data);
    }
    const cvm::Op *op = cvm::Op::Get("conv2d");
    static auto& fextra_space =
        Op::GetAttr<cvm::FOpExtraSpace>("FOpExtraSpace");
    auto fextra = fextra_space.get(op, nullptr);
    static auto& finfer_shape =
        Op::GetAttr<cvm::FInferNodeEntryAttr<TShape> >("FInferShape");
    auto finfer = finfer_shape.get(op, nullptr);
    int64_t es[1]{ 0 };
    CVMOpParam params;
    params.func_name = "conv2d";
    if(bias)
        params.num_inputs = 3;
    else
        params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;
    vector<TShape> ishape(params.num_inputs), oshape(params.num_outputs);
    vector<DLTensor> args(params.num_inputs + params.num_outputs);
    DLTensor* cpu_tensor;
    for(int i = 0; i < ishape.size(); i++)
    {
        TShape shp(tshape[i].size());
        for(int ti=0;ti<shp.ndim();ti++)
            shp[ti] = tshape[i][ti];
        ishape[i] = shp;
        DLTensor* dl;
        CVMArrayAlloc((int64_t*)tshape[i].data(), tshape[i].size(), dtype_code, dtype_bits, dtype_lanes, ctx, device_id, &dl);
        args[i] = *dl;
        CVMArrayAlloc((int64_t*)tshape[i].data(), tshape[i].size(), dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
        memcpy(cpu_tensor->data, tdata[i].data(), sizeof(int32_t) * tdata[i].size());
        CVMArrayCopyFromTo(cpu_tensor, dl, nullptr);
        CVMArrayFree(cpu_tensor);
    }
    NodeAttrs attr;
    LoadOp("conv2d", attr);
    LoadOpAttr(attr_str, attr);
    finfer(attr, &ishape, &oshape);
    vector<int> out_shape;
    for(int i=0;i<oshape.size();i++)
    {
        out_shape.push_back(oshape[0][i]);
    }
    tshape.push_back(out_shape);
    DLTensor* dl;
    CVMArrayAlloc((int64_t*)tshape[params.num_inputs].data(), tshape[params.num_inputs].size(), dtype_code, dtype_bits, dtype_lanes, ctx, device_id, &dl);
    args[params.num_inputs] = *dl;
    vector<int> iprecs;
    if (fextra != nullptr)
    {
        es[0] = fextra(attr, &ishape, &iprecs,
                       DLContext{DLDeviceType(ctx), 0});
    }
    DLTensor* extra_space;
    CVMArrayAlloc(es, 1, 
        dtype_code, dtype_bits, dtype_lanes, ctx, device_id, &extra_space);
    auto op_slice = get_func(params, &attr, args, params.num_inputs, extra_space);
    op_slice();
    int out_size = out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3];
    vector<int> cpu_output_tensor(out_size);
    {
        DLTensor* cpu_tensor;
        int i = params.num_inputs;
        CVMArrayAlloc((int64_t*)tshape[i].data(), tshape[i].size(), dtype_code,
                      dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
        CVMArrayCopyFromTo(&args[i], cpu_tensor, nullptr);
        memcpy(cpu_output_tensor.data(), cpu_tensor->data,
               sizeof(int32_t) * out_size);
        CVMArrayFree(cpu_tensor);
    }
    cout << "out: " <<endl;
    for(int i = 0;i<out_size;i++)
    {
        cout<< cpu_output_tensor[i]<<" ";
    }
    cout<<endl;
    cout<< "out_shape: "<<endl;
    for(int i=0;i< out_shape.size();i++)
    {
        cout<< out_shape[i] << " ";
    }
    cout<<endl;
}

void dense(vector<vector<int>> m1, vector<vector<int>> m2, vector<vector<int>> output, string attr_str, DLTensor* extra_space ,vector<int> bias = {})
{
    CVMOpParam params;
    params.func_name = "dense";
    if (bias.size() == 0)
        params.num_inputs = 2;
    else
        params.num_inputs = 3;
    params.num_outputs = 1;
    vector<DLTensor> args(params.num_inputs + params.num_outputs);
    vector<vector<uint64_t>> tshape(args.size());
    vector<vector<int32_t>> tdata(args.size());

    for (int i = 0; i < m1.size(); i++)
        for (int j = 0; j < m1[0].size(); j++)
            tdata[0].push_back(m1[i][j]);
    for (int i = 0; i < m2.size(); i++)
        for (int j = 0; j < m2[0].size(); j++)
            tdata[0].push_back(m2[i][j]);
    for (int i = 0; i < output.size(); i++)
        for (int j = 0; j < output[0].size(); j++)
            tdata[0].push_back(output[i][j]);
    tshape[0] = {m1.size(), m1[0].size()};
    tshape[1] = {m2.size(), m2[0].size()};
    tshape[params.num_inputs] = {output.size(), output[0].size()};
    if (params.num_inputs == 3)
    {
        tdata[2] = bias;
        tshape[2] = {bias.size()};
    }
    handle(tshape, tdata, args, params, attr_str, extra_space);
}

void max_pool2d(vector<vector<vector<vector<int>>>> x, vector<vector<vector<vector<int>>>> y, string attr_str, DLTensor* extra_space)
{
    CVMOpParam params;
    params.func_name = "max_pool2d";
    params.num_inputs = 1;
    params.num_outputs = 1;
    vector<DLTensor> args(params.num_inputs + params.num_outputs);
    vector<vector<uint64_t>> tshape(args.size());
    vector<vector<int32_t>> tdata(args.size());

    for (int i = 0; i < x.size(); i++)
        for (int j = 0; j < x[0].size(); j++)
            for (int k = 0; k < x[0][0].size(); k++)
                for (int l = 0; l < x[0][0][0].size(); l++)
                    tdata[0].push_back(x[i][j][k][l]);
    for (int i = 0; i < y.size(); i++)
        for (int j = 0; j < y[0].size(); j++)
            for (int k = 0; k < y[0][0].size(); k++)
                for (int l = 0; l < y[0][0][0].size(); l++)
                    tdata[1].push_back(y[i][j][k][l]);

    tshape[0] = {x.size(), x[0].size(), x[0][0].size(), x[0][0][0].size()};
    tshape[1] = {y.size(), y[0].size(), y[0][0].size(), y[0][0][0].size()};
    handle(tshape, tdata, args, params, attr_str, extra_space);
}

void upsampling(vector<vector<vector<vector<int>>>> x, vector<vector<vector<vector<int>>>> y, string attr_str, DLTensor* extra_space)
{
    CVMOpParam params;
    params.func_name = "upsampling";
    params.num_inputs = 1;
    params.num_outputs = 1;
    vector<DLTensor> args(params.num_inputs + params.num_outputs);
    vector<vector<uint64_t>> tshape(args.size());
    vector<vector<int32_t>> tdata(args.size());

    for (int i = 0; i < x.size(); i++)
        for (int j = 0; j < x[0].size(); j++)
            for (int k = 0; k < x[0][0].size(); k++)
                for (int l = 0; l < x[0][0][0].size(); l++)
                    tdata[0].push_back(x[i][j][k][l]);
    for (int i = 0; i < y.size(); i++)
        for (int j = 0; j < y[0].size(); j++)
            for (int k = 0; k < y[0][0].size(); k++)
                for (int l = 0; l < y[0][0][0].size(); l++)
                    tdata[1].push_back(y[i][j][k][l]);

    tshape[0] = {x.size(), x[0].size(), x[0][0].size(), x[0][0][0].size()};
    tshape[1] = {y.size(), y[0].size(), y[0][0].size(), y[0][0][0].size()};
    handle(tshape, tdata, args, params, attr_str, extra_space);
}

void get_valid_counts(vector<vector<vector<int>>> x, vector<int> out0, vector<vector<vector<int>>> out1, string attr_str, DLTensor* extra_space)
{
    CVMOpParam params;
    params.func_name = "get_valid_counts";
    params.num_inputs = 1;
    params.num_outputs = 2;
    vector<DLTensor> args(params.num_inputs + params.num_outputs);
    vector<vector<uint64_t>> tshape(args.size());
    vector<vector<int32_t>> tdata(args.size());

    for (int i = 0; i < x.size(); i++)
        for (int j = 0; j < x[0].size(); j++)
            for (int k = 0; k < x[0][0].size(); k++)
                tdata[0].push_back(x[i][j][k]);
    for (int i = 0; i < out0.size(); i++)
        tdata[1].push_back(out0[i]);
    for (int i = 0; i < out1.size(); i++)
        for (int j = 0; j < out1[0].size(); j++)
            for (int k = 0; k < out1[0][0].size(); k++)
                tdata[2].push_back(out1[i][j][k]);
    tshape[0] = {x.size(), x[0].size(), x[0][0].size()};
    tshape[1] = {out0.size()};
    tshape[2] = {out1.size(), out1[0].size(), out1[0][0].size()};
    handle(tshape, tdata, args, params, attr_str, extra_space);
}

void test(string op_name)
{
    vector<string> case_list;
    string case_dir = CASE_DIR + "/" + op_name + "/";
    findAllSubDir(case_list, case_dir.c_str(), TYPE_DIR);

    static auto& finfer_shape = 
        Op::GetAttr<cvm::FInferNodeEntryAttr<TShape> >("FInferShape");
    const cvm::Op *op = cvm::Op::Get(op_name);
    auto finfer = finfer_shape.get(op, nullptr);
    if (finfer == nullptr) {
        std::cout << "operator " << op_name
        << "has not registered FInferShape";
        return ;
    }

    static auto& finfer_prec =
        Op::GetAttr<cvm::FInferPrecision>("FInferPrecision");
    auto fip = finfer_prec.get(op, nullptr);
    if (fip == nullptr) {
        std::cout << "operator " << op_name
        << "has not registered FInferPrecision";
        return ;
    }

    static auto& fextra_space =
        Op::GetAttr<cvm::FOpExtraSpace >("FOpExtraSpace");
    auto fextra = fextra_space.get(op, nullptr);

    for(unsigned int ci = 0; ci < case_list.size(); ci++)
    {
        string case_path = case_dir + case_list[ci] + "/";
        cout << "doing test case at " << case_path << endl;
        string attr_path = case_path + "attr.txt";
        string attr_str = "";
        read_one_line(attr_path, attr_str);
        std::cout << attr_str << endl;
        vector<string> file_list;
        findAllSubDir(file_list, case_path.c_str(), TYPE_FILE);
        unsigned int num_inputs = 0, num_outputs = 0;  // num_outputs is the expected number of output
        for(auto file_name : file_list)
        {
            if(file_name.find("in_") != string::npos)
                num_inputs += 1;
            if(file_name.find("out_") != string::npos)
                num_outputs += 1;
        }
        NodeAttrs attr;
        LoadOp(op_name, attr);
        LoadOpAttr(attr_str, attr);

        CVMOpParam params;
        params.func_name = op_name;
        params.num_inputs = num_inputs;
        params.num_outputs =
            attr.op->get_num_outputs ? attr.op->get_num_outputs(attr) : attr.op->num_outputs;
        params.flatten_data = false;
        vector<DLTensor> args(params.num_inputs + params.num_outputs);
        vector<vector<unsigned long>> tshape(args.size());
        vector<vector<int32_t>> tdata(args.size());
        vector<TShape> ishape(params.num_inputs), oshape(params.num_outputs);
        for(int i = 0; i < num_inputs; i++)
        {
            string in_path = case_path + "in_" + to_string(i) + ".txt";
            read_data(in_path.c_str(), tshape[i], tdata[i]);
            TShape shp(tshape[i].size());
            for(uint ti = 0; ti < shp.ndim(); ++ti)
            {
                shp[ti] = tshape[i][ti];
            }
            ishape[i] = shp;
            cout << shp << endl;
        }

        bool infer_shape_ret;
        string err_path = case_path + "err.txt", err_str = "";
        read_one_line(err_path, err_str);
        try
        {
            infer_shape_ret = finfer(attr, &ishape, &oshape);
            if (infer_shape_ret)
            {
                cout << "FInferShape ishape=[";
                for (auto &shp : ishape)
                    cout << shp << ", ";
                cout << "] oshape=[";
                for (auto &shp : oshape)
                    cout << shp << ", ";
                cout << "]\n";
            }
        }
        catch (const exception &e)
        {
            cerr << "FInferShape error with " << e.what() << endl;
            infer_shape_ret = false;
        }
        if (infer_shape_ret == false)
        {
            cout << "error after FInferShape: " << err_str << endl;
            if (err_str == "")
            {
                string out_path = case_path + "out_0.txt";
                cout << out_path << endl;
                read_data(out_path.c_str(), tshape[num_inputs], tdata[num_inputs]);
                print(tdata[num_inputs]);
                assert(false);
                continue;
            }
            else
            {
                cout << "match 1 | 0" << endl;
                continue;
            }
        }
        vector<int> iprecs;
        int64_t es[1]{0};
        if (fextra != nullptr)
        {
            es[0] = fextra(attr, &ishape, &iprecs, DLContext{DLDeviceType(ctx), 0});
        }
        DLTensor* extra_space;
        CVMArrayAlloc(es, 1, 
            dtype_code, dtype_bits, dtype_lanes, ctx, device_id, &extra_space);
        if (params.num_outputs != num_outputs)
        {
            cout << "error with FInfershape or op.get_num_outputs()\n"
                 << "should be " << num_outputs << " outputs but we calculated "
                 << params.num_outputs << endl;
            assert(false);
        }
        for(int j = 0; j < num_outputs; j++)
        {
            string out_path = case_path + "out_" + to_string(j) + ".txt";
            read_data(out_path.c_str(), tshape[num_inputs + j], tdata[num_inputs + j]);
            int shape_cmp = memcmp(tshape[num_inputs + j].data(), oshape[j].data(), sizeof(int64_t) * tshape[num_inputs+j].size());
            if(shape_cmp != 0)
            {
                print(tshape[num_inputs+j]);
                std::cout << "oshape[" << j << "] = " << oshape[j] << endl;
            }
            assert(shape_cmp == 0);
        }
        handle(tshape, tdata, args, params, attr_str, extra_space);
    }
}

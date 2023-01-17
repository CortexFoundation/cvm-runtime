#include "test_op.cc"

struct data_and_shape
{
    int32_t* data;
    vector<int64_t> shape;
};

vector<NDArray> result(vector<vector<int64_t>> tshape, vector<vector<int32_t>> tdata, CVMOpParam params, NodeAttrs attr)
{
    const cvm::Op *op = cvm::Op::Get(params.func_name);
    static auto &fextra_space =
        Op::GetAttr<cvm::FOpExtraSpace>("FOpExtraSpace");
    auto fextra = fextra_space.get(op, nullptr);
    static auto &finfer_shape =
        Op::GetAttr<cvm::FInferNodeEntryAttr<TShape>>("FInferShape");
    auto finfer = finfer_shape.get(op, nullptr);
    int64_t es[1]{0};
    vector<TShape> ishape(params.num_inputs), oshape(params.num_outputs);
    vector<DLTensor> args(params.num_inputs + params.num_outputs);
    DLTensor *cpu_tensor;
    for (int i = 0; i < ishape.size(); i++)
    {
        TShape shp(tshape[i].size());
        for (int ti = 0; ti < shp.ndim(); ti++)
            shp[ti] = tshape[i][ti];
        ishape[i] = shp;
        DLTensor *dl;
        CVMArrayAlloc(tshape[i].data(), tshape[i].size(), dtype_code, dtype_bits, dtype_lanes, ctx, device_id, &dl);
        args[i] = *dl;
        CVMArrayAlloc(tshape[i].data(), tshape[i].size(), dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
        memcpy(cpu_tensor->data, tdata[i].data(), sizeof(int32_t) * tdata[i].size());
        CVMArrayCopyFromTo(cpu_tensor, dl, nullptr);
        CVMArrayFree(cpu_tensor);
    }
    finfer(attr, &ishape, &oshape);
    for (int i = 0; i < oshape.size(); i++)
    {
        vector<int64_t> out_shape;
        for (int j = 0; j < oshape[j].ndim(); j++)
        {
            out_shape.push_back(oshape[i][j]);
        }
        tshape.push_back(out_shape);
    }
    for (int i = 0; i < params.num_outputs; i++)
    {
        DLTensor *dl;
        CVMArrayAlloc(tshape[params.num_inputs + i].data(), tshape[params.num_inputs + i].size(), dtype_code, dtype_bits, dtype_lanes, ctx, device_id, &dl);
        args[params.num_inputs + i] = *dl;
    }
    vector<int> iprecs;
    if (fextra != nullptr)
    {
        es[0] = fextra(attr, &ishape, &iprecs,
                       DLContext{DLDeviceType(ctx), 0});
    }
    DLTensor *extra_space;
    CVMArrayAlloc(es, 1,
                  dtype_code, dtype_bits, dtype_lanes, ctx, device_id, &extra_space);
    auto op_slice = get_func(params, &attr, args, params.num_inputs, extra_space);
    op_slice();
    vector<NDArray> res(params.num_outputs);
    for (int out_no = 0; out_no < params.num_outputs; out_no++)
    {
        int out_size = 1;
        for (int i = 0; i < tshape[params.num_inputs + out_no].size(); i++)
        {
            out_size *= tshape[params.num_inputs + out_no][i];
        }
        vector<int32_t> cpu_output_tensor(out_size);
        {
            DLTensor *cpu_tensor;
            int i = params.num_inputs;
            CVMArrayAlloc(tshape[i + out_no].data(), tshape[i + out_no].size(), dtype_code,
                          dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
            CVMArrayCopyFromTo(&args[i + out_no], cpu_tensor, nullptr);
            memcpy(cpu_output_tensor.data(), cpu_tensor->data,
                   sizeof(int32_t) * out_size);
            CVMArrayFree(cpu_tensor);
        }
        DLDataType dtype = {dtype_code, dtype_bits, dtype_lanes};
        DLContext dlctx = {APIDevTypeMap.at(ctx), 0};
        NDArray ret = NDArray::Empty(tshape[params.num_inputs + out_no], dtype, dlctx);
        for (int i = 0; i < out_size; i++)
        {
            static_cast<int32_t *>(const_cast<DLTensor *>(ret.operator->())->data)[i] = cpu_output_tensor[i];
        }
        CVMArrayFree(extra_space);
        res[out_no] = ret;
    }
    return res;
}


// NDArray
NDArray conv2d(NDArray const &data, NDArray const &weight, TShape padding, TShape strides, TShape dilation, int groups, NDArray *bias = nullptr)
{
    CVMOpParam params;
    params.func_name = "conv2d";
    if (bias)
        params.num_inputs = 3;
    else
        params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // data
    int32_t *ddata = static_cast<int32_t *>(data->data);
    vector<int64_t> dshape = vector<int64_t>(data->shape, data->shape + data->ndim);
    tshape[0] = dshape;
    int dsize = 1;
    for (int i = 0; i < data->ndim; i++)
    {
        dsize *= dshape[i];
    }
    vector<int32_t> d_data(dsize);
    for (int i = 0; i < dsize; i++)
    {
        d_data[i] = ddata[i];
    }
    tdata[0] = d_data;
    // weight
    vector<int64_t> wshape = vector<int64_t>(weight->shape, weight->shape + weight->ndim);
    tshape[1] = wshape;
    int32_t *wdata = static_cast<int32_t *>(weight->data);
    int wsize = 1;
    for (int i = 0; i < weight->ndim; i++)
    {
        wsize *= wshape[i];
    }
    vector<int32_t> w_data(wsize);
    for (int i = 0; i < wsize; i++)
    {
        w_data[i] = wdata[i];
    }
    tdata[1] = w_data;
    // bias
    if (bias != nullptr)
    {
        DLTensor *bptr = bias->operator->();
        vector<int64_t> bshape = vector<int64_t>(bptr->shape, bptr->shape + bptr->ndim);
        tshape[2] = bshape;
        int32_t *bdata = static_cast<int32_t *>(bptr->data);
        int bsize = bshape[0];
        vector<int32_t> b_data(bsize);
        for (int i = 0; i < bsize; i++)
        {
            b_data[i] = bdata[i];
        }
        tdata[2] = b_data;
    }

    string use_bias = bias == nullptr ? "False" : "True";
    string attr_str = "{\"channels\":" + to_string(dshape[0]) + "\", \"kernel_size\":\"[" + to_string(dshape[2]) + ", " + to_string(dshape[3]) +
                      "], \"strides\":\"[" + to_string(strides[0]) + ", " + to_string(strides[1]) + "]\", \"padding\":\"[" + to_string(padding[0]) +
                      ", " + to_string(padding[1]) + "]\", \"dilation\":\"(" + to_string(dilation[0]) + ", " + to_string(dilation[1]) + ")\", \"groups\":\"" +
                      to_string(groups) + "\", \"layout\":\"NCHW\", \"kernel_layout\":\"OIHW\", \"use_bias\":\"" + use_bias + "\"}";
    NodeAttrs attr;
    LoadOp("conv2d", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray dense(NDArray const &data, NDArray const &weight, NDArray *bias = nullptr)
{
    CVMOpParam params;
    params.func_name = "dense";
    if (bias)
        params.num_inputs = 3;
    else
        params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // data
    int32_t *ddata = static_cast<int32_t *>(data->data);
    vector<int64_t> dshape = vector<int64_t>(data->shape, data->shape + data->ndim);
    tshape[0] = dshape;
    int dsize = 1;
    for (int i = 0; i < data->ndim; i++)
    {
        dsize *= dshape[i];
    }
    vector<int32_t> d_data(dsize);
    for (int i = 0; i < dsize; i++)
    {
        d_data[i] = ddata[i];
    }
    tdata[0] = d_data;
    // weight
    vector<int64_t> wshape = vector<int64_t>(weight->shape, weight->shape + weight->ndim);
    tshape[1] = wshape;
    int32_t *wdata = static_cast<int32_t *>(weight->data);
    int wsize = 1;
    for (int i = 0; i < weight->ndim; i++)
    {
        wsize *= wshape[i];
    }
    vector<int32_t> w_data(wsize);
    for (int i = 0; i < wsize; i++)
    {
        w_data[i] = wdata[i];
    }
    tdata[1] = w_data;
    // bias
    if (bias != nullptr)
    {
        DLTensor *bptr = bias->operator->();
        vector<int64_t> bshape = vector<int64_t>(bptr->shape, bptr->shape + bptr->ndim);
        tshape[2] = bshape;
        int32_t *bdata = static_cast<int32_t *>(bptr->data);
        int bsize = bshape[0];
        vector<int32_t> b_data(bsize);
        for (int i = 0; i < bsize; i++)
        {
            b_data[i] = bdata[i];
        }
        tdata[2] = b_data;
    }

    string use_bias = bias == nullptr ? "False" : "True";
    string attr_str = "\"units\":\"" + to_string(wshape[0]) + "\", \"use_bias\":\"" + use_bias + "\"}";
    NodeAttrs attr;
    LoadOp("dense", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray max_pool2d(NDArray const &x, TShape pool_size, TShape strides, TShape padding, bool ceil_mode)
{
    CVMOpParam params;
    params.func_name = "max_pool2d";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string ceil = ceil_mode == true ? "True" : "False";
    string attr_str = "{\"pool_size\":\"" + to_string(pool_size[0]) + ", " + to_string(pool_size[1]) +
                      "]\", \"strides\":\"[" + to_string(padding[0]) + ", " + to_string(padding[1]) + "]\", \"ceil_mode\":\"" +
                      ceil + "\"}";
    NodeAttrs attr;
    LoadOp("max_pool2d", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray upsampling(NDArray const &x, int scale)
{
    CVMOpParam params;
    params.func_name = "upsampling";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string attr_str = "{\"scale\":" + to_string(scale) + "\"}";
    NodeAttrs attr;
    LoadOp("upsampling", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

vector<NDArray> get_valid_counts(NDArray const &x, int score_threshold)
{
    CVMOpParam params;
    params.func_name = "get_valid_counts";
    params.num_inputs = 1;
    params.num_outputs = 2;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = xshape[0] * xshape[1] * xshape[2];
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string attr_str = "{\"score_threshold\":\"" + to_string(score_threshold) + "\"}";
    NodeAttrs attr;
    LoadOp("get_valid_counts", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr);
}

NDArray sum(NDArray const &x, TShape axis, bool keepdims, bool exclude)
{
    CVMOpParam params;
    params.func_name = "sum";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string kdims = keepdims == true ? "True" : "False";
    string exc = exclude == true ? "True" : "False";
    string attr_str = "{\"axis\":\"[" + to_string(axis[0]) + ", " + to_string(axis[1]) + ", " +
                      to_string(axis[2]) + "]\", \"keepdims\":\"" + kdims + "\", \"exclude\":\"" + exc + "\"}";
    NodeAttrs attr;
    LoadOp("sum", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray max(NDArray const &x, TShape axis, bool keepdims, bool exclude)
{
    CVMOpParam params;
    params.func_name = "max";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string kdims = keepdims == true ? "True" : "False";
    string exc = exclude == true ? "True" : "False";
    string attr_str = "{\"axis\":\"[" + to_string(axis[0]) + ", " + to_string(axis[1]) + ", " +
                      to_string(axis[2]) + "]\", \"keepdims\":\"" + kdims + "\", \"exclude\":\"" + exc + "\"}";
    NodeAttrs attr;
    LoadOp("max", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray broadcast_add(NDArray const &x, NDArray const &y)
{
    CVMOpParam params;
    params.func_name = "broadcast_add";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_add", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray broadcast_div(NDArray const &x, NDArray const &y)
{
    CVMOpParam params;
    params.func_name = "broadcast_div";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_div", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray broadcast_greater(NDArray const &x, NDArray const &y)
{
    CVMOpParam params;
    params.func_name = "broadcast_greater";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_greater", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray broadcast_max(NDArray const &x, NDArray const &y)
{
    CVMOpParam params;
    params.func_name = "broadcast_max";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_max", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray broadcast_mul(NDArray const &x, NDArray const &y)
{
    CVMOpParam params;
    params.func_name = "broadcast_mul";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_mul", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray broadcast_sub(NDArray const &x, NDArray const &y)
{
    CVMOpParam params;
    params.func_name = "broadcast_sub";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_sub", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray slice_like(NDArray const &x, NDArray const &y, TShape axis)
{
    CVMOpParam params;
    params.func_name = "slice_like";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{\"axis\":\"[" + to_string(axis[0]) + ", " + to_string(axis[1]) + ", " + to_string(axis[2]) + ", " + to_string(axis[3]) + "]\"";
    NodeAttrs attr;
    LoadOp("slice_like", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray tile(NDArray const &x, NDArray const &y, TShape reps)
{
    CVMOpParam params;
    params.func_name = "tile";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{\"reps\":\"[";
    for (int i = 0; i < reps.ndim() - 1; i++)
    {
        attr_str += to_string(reps[i]) + ", ";
    }
    attr_str += to_string(reps[reps.ndim() - 1]) + "]\"}";
    NodeAttrs attr;
    LoadOp("tile", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray repeat(NDArray const &x, int repeats, int axis)
{
    CVMOpParam params;
    params.func_name = "repeat";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string attr_str = "{\"repeats\":\"" + to_string(repeats) + "\", \"axis\":\"" + to_string(axis) + "\"}";
    NodeAttrs attr;
    LoadOp("repeat", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray strided_slice(NDArray const &x, TShape begin, TShape end, TShape stride)
{
    CVMOpParam params;
    params.func_name = "strided_slice";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string attr_str = "{\"begin\":\"[";
    for (int i = 0; i < begin.ndim() - 1; i++)
        attr_str += to_string(begin[i]) + ", ";
    attr_str += to_string(begin[begin.ndim() - 1]) + "]\", \"end\":\"[";
    for (int i = 0; i < end.ndim() - 1; i++)
        attr_str += to_string(end[i]) + ", ";
    attr_str += to_string(end[end.ndim() - 1]) + "]\", \"stride\":\"[";
    for (int i = 0; i < stride.ndim() - 1; i++)
        attr_str += to_string(stride[i]) + ", ";
    attr_str += to_string(stride[stride.ndim() - 1]) + "]\"}";
    NodeAttrs attr;
    LoadOp("strided_slice", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray concatenate(vector<NDArray> const &x, int axis)
{
    CVMOpParam params;
    params.func_name = "concatenate";
    params.num_inputs = x.size();
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    for (int in = 0; in < params.num_inputs; in++)
    {
        int32_t *data = static_cast<int32_t *>(x[in]->data);
        vector<int64_t> shape = vector<int64_t>(x[in]->shape, x[in]->shape + x[in]->ndim);
        tshape[in] = shape;
        int in_size = 1;
        for (int i = 0; i < x[in]->ndim; i++)
        {
            in_size *= shape[i];
        }
        vector<int32_t> x_data(in_size);
        for (int i = 0; i < in_size; i++)
        {
            x_data[i] = data[i];
        }
        tdata[in] = x_data;
    }

    string attr_str = "{\"axis\":\"" + to_string(axis) + "\"}";
    NodeAttrs attr;
    LoadOp("concatenate", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray transpose(NDArray const &x, TShape axes)
{
    CVMOpParam params;
    params.func_name = "transpose";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string attr_str = "{\"axes\":\"[";
    for (int i = 0; i < axes.ndim() - 1; i++)
    {
        attr_str += to_string(axes[i]) + ", ";
    }
    attr_str += to_string(axes[axes.ndim() - 1]) + "]\"}";
    NodeAttrs attr;
    LoadOp("transpose", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}

NDArray take(NDArray const &x, NDArray const &y, int axis)
{
    CVMOpParam params;
    params.func_name = "take";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{\"axis\":\"" + to_string(axis) + "\"}";
    NodeAttrs attr;
    LoadOp("take", attr);
    LoadOpAttr(attr_str, attr);
    return result(tshape, tdata, params, attr)[0];
}


// DLTensor
DLTensor *conv2d(DLTensor *data, DLTensor *weight, TShape padding, TShape strides, TShape dilation, int groups, DLTensor *bias = nullptr)
{
    CVMOpParam params;
    params.func_name = "conv2d";
    if (bias)
        params.num_inputs = 3;
    else
        params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // data
    int32_t *ddata = static_cast<int32_t *>(data->data);
    vector<int64_t> dshape = vector<int64_t>(data->shape, data->shape + data->ndim);
    tshape[0] = dshape;
    int dsize = 1;
    for (int i = 0; i < data->ndim; i++)
    {
        dsize *= dshape[i];
    }
    vector<int32_t> d_data(dsize);
    for (int i = 0; i < dsize; i++)
    {
        d_data[i] = ddata[i];
    }
    tdata[0] = d_data;
    // weight
    vector<int64_t> wshape = vector<int64_t>(weight->shape, weight->shape + weight->ndim);
    tshape[1] = wshape;
    int32_t *wdata = static_cast<int32_t *>(weight->data);
    int wsize = 1;
    for (int i = 0; i < weight->ndim; i++)
    {
        wsize *= wshape[i];
    }
    vector<int32_t> w_data(wsize);
    for (int i = 0; i < wsize; i++)
    {
        w_data[i] = wdata[i];
    }
    tdata[1] = w_data;
    // bias
    if (bias != nullptr)
    {
        vector<int64_t> bshape = vector<int64_t>(bias->shape, bias->shape + bias->ndim);
        tshape[2] = bshape;
        int32_t *bdata = static_cast<int32_t *>(bias->data);
        int bsize = bshape[0];
        vector<int32_t> b_data(bsize);
        for (int i = 0; i < bsize; i++)
        {
            b_data[i] = bdata[i];
        }
        tdata[2] = b_data;
    }

    string use_bias = bias == nullptr ? "False" : "True";
    string attr_str = "{\"channels\":" + to_string(dshape[0]) + "\", \"kernel_size\":\"[" + to_string(dshape[2]) + ", " + to_string(dshape[3]) +
                      "], \"strides\":\"[" + to_string(strides[0]) + ", " + to_string(strides[1]) + "]\", \"padding\":\"[" + to_string(padding[0]) +
                      ", " + to_string(padding[1]) + "]\", \"dilation\":\"(" + to_string(dilation[0]) + ", " + to_string(dilation[1]) + ")\", \"groups\":\"" +
                      to_string(groups) + "\", \"layout\":\"NCHW\", \"kernel_layout\":\"OIHW\", \"use_bias\":\"" + use_bias + "\"}";
    NodeAttrs attr;
    LoadOp("conv2d", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *dense(DLTensor *data, DLTensor *weight, NDArray *bias = nullptr)
{
    CVMOpParam params;
    params.func_name = "dense";
    if (bias)
        params.num_inputs = 3;
    else
        params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // data
    int32_t *ddata = static_cast<int32_t *>(data->data);
    vector<int64_t> dshape = vector<int64_t>(data->shape, data->shape + data->ndim);
    tshape[0] = dshape;
    int dsize = 1;
    for (int i = 0; i < data->ndim; i++)
    {
        dsize *= dshape[i];
    }
    vector<int32_t> d_data(dsize);
    for (int i = 0; i < dsize; i++)
    {
        d_data[i] = ddata[i];
    }
    tdata[0] = d_data;
    // weight
    vector<int64_t> wshape = vector<int64_t>(weight->shape, weight->shape + weight->ndim);
    tshape[1] = wshape;
    int32_t *wdata = static_cast<int32_t *>(weight->data);
    int wsize = 1;
    for (int i = 0; i < weight->ndim; i++)
    {
        wsize *= wshape[i];
    }
    vector<int32_t> w_data(wsize);
    for (int i = 0; i < wsize; i++)
    {
        w_data[i] = wdata[i];
    }
    tdata[1] = w_data;
    // bias
    if (bias != nullptr)
    {
        DLTensor *bptr = bias->operator->();
        vector<int64_t> bshape = vector<int64_t>(bptr->shape, bptr->shape + bptr->ndim);
        tshape[2] = bshape;
        int32_t *bdata = static_cast<int32_t *>(bptr->data);
        int bsize = bshape[0];
        vector<int32_t> b_data(bsize);
        for (int i = 0; i < bsize; i++)
        {
            b_data[i] = bdata[i];
        }
        tdata[2] = b_data;
    }

    string use_bias = bias == nullptr ? "False" : "True";
    string attr_str = "\"units\":\"" + to_string(wshape[0]) + "\", \"use_bias\":\"" + use_bias + "\"}";
    NodeAttrs attr;
    LoadOp("dense", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *max_pool2d(DLTensor *x, TShape pool_size, TShape strides, TShape padding, bool ceil_mode)
{
    CVMOpParam params;
    params.func_name = "max_pool2d";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string ceil = ceil_mode == true ? "True" : "False";
    string attr_str = "{\"pool_size\":\"" + to_string(pool_size[0]) + ", " + to_string(pool_size[1]) +
                      "]\", \"strides\":\"[" + to_string(padding[0]) + ", " + to_string(padding[1]) + "]\", \"ceil_mode\":\"" +
                      ceil + "\"}";
    NodeAttrs attr;
    LoadOp("max_pool2d", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *upsampling(DLTensor *x, int scale)
{
    CVMOpParam params;
    params.func_name = "upsampling";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string attr_str = "{\"scale\":" + to_string(scale) + "\"}";
    NodeAttrs attr;
    LoadOp("upsampling", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

vector<DLTensor *> get_valid_counts(DLTensor *x, int score_threshold)
{
    CVMOpParam params;
    params.func_name = "get_valid_counts";
    params.num_inputs = 1;
    params.num_outputs = 2;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = xshape[0] * xshape[1] * xshape[2];
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string attr_str = "{\"score_threshold\":\"" + to_string(score_threshold) + "\"}";
    NodeAttrs attr;
    LoadOp("get_valid_counts", attr);
    LoadOpAttr(attr_str, attr);
    vector<NDArray> res = result(tshape, tdata, params, attr);
    vector<DLTensor *> out(res.size());
    for (int i = 0; i < res.size(); i++)
    {
        out[i] = res[i].operator->();
    }
    return out;
}

DLTensor *sum(DLTensor *x, TShape axis, bool keepdims, bool exclude)
{
    CVMOpParam params;
    params.func_name = "sum";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string kdims = keepdims == true ? "True" : "False";
    string exc = exclude == true ? "True" : "False";
    string attr_str = "{\"axis\":\"[" + to_string(axis[0]) + ", " + to_string(axis[1]) + ", " +
                      to_string(axis[2]) + "]\", \"keepdims\":\"" + kdims + "\", \"exclude\":\"" + exc + "\"}";
    NodeAttrs attr;
    LoadOp("sum", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *max(DLTensor *x, TShape axis, bool keepdims, bool exclude)
{
    CVMOpParam params;
    params.func_name = "max";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string kdims = keepdims == true ? "True" : "False";
    string exc = exclude == true ? "True" : "False";
    string attr_str = "{\"axis\":\"[" + to_string(axis[0]) + ", " + to_string(axis[1]) + ", " +
                      to_string(axis[2]) + "]\", \"keepdims\":\"" + kdims + "\", \"exclude\":\"" + exc + "\"}";
    NodeAttrs attr;
    LoadOp("max", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *broadcast_add(DLTensor *x, DLTensor *y)
{
    CVMOpParam params;
    params.func_name = "broadcast_add";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_add", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *broadcast_div(DLTensor *x, DLTensor *y)
{
    CVMOpParam params;
    params.func_name = "broadcast_div";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_div", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *broadcast_greater(DLTensor *x, DLTensor *y)
{
    CVMOpParam params;
    params.func_name = "broadcast_greater";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_greater", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *broadcast_max(DLTensor *x, DLTensor *y)
{
    CVMOpParam params;
    params.func_name = "broadcast_max";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_max", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *broadcast_mul(DLTensor *x, DLTensor *y)
{
    CVMOpParam params;
    params.func_name = "broadcast_mul";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_mul", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *broadcast_sub(DLTensor *x, DLTensor *y)
{
    CVMOpParam params;
    params.func_name = "broadcast_sub";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_sub", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *slice_like(DLTensor *x, DLTensor *y, TShape axis)
{
    CVMOpParam params;
    params.func_name = "slice_like";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{\"axis\":\"[" + to_string(axis[0]) + ", " + to_string(axis[1]) + ", " + to_string(axis[2]) + ", " + to_string(axis[3]) + "]\"";
    NodeAttrs attr;
    LoadOp("slice_like", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *tile(DLTensor *x, DLTensor *y, TShape reps)
{
    CVMOpParam params;
    params.func_name = "tile";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{\"reps\":\"[";
    for (int i = 0; i < reps.ndim() - 1; i++)
    {
        attr_str += to_string(reps[i]) + ", ";
    }
    attr_str += to_string(reps[reps.ndim() - 1]) + "]\"}";
    NodeAttrs attr;
    LoadOp("tile", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *repeat(DLTensor *x, int repeats, int axis)
{
    CVMOpParam params;
    params.func_name = "repeat";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string attr_str = "{\"repeats\":\"" + to_string(repeats) + "\", \"axis\":\"" + to_string(axis) + "\"}";
    NodeAttrs attr;
    LoadOp("repeat", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *strided_slice(DLTensor *x, TShape begin, TShape end, TShape stride)
{
    CVMOpParam params;
    params.func_name = "strided_slice";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string attr_str = "{\"begin\":\"[";
    for (int i = 0; i < begin.ndim() - 1; i++)
        attr_str += to_string(begin[i]) + ", ";
    attr_str += to_string(begin[begin.ndim() - 1]) + "]\", \"end\":\"[";
    for (int i = 0; i < end.ndim() - 1; i++)
        attr_str += to_string(end[i]) + ", ";
    attr_str += to_string(end[end.ndim() - 1]) + "]\", \"stride\":\"[";
    for (int i = 0; i < stride.ndim() - 1; i++)
        attr_str += to_string(stride[i]) + ", ";
    attr_str += to_string(stride[stride.ndim() - 1]) + "]\"}";
    NodeAttrs attr;
    LoadOp("strided_slice", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *concatenate(vector<DLTensor *> x, int axis)
{
    CVMOpParam params;
    params.func_name = "concatenate";
    params.num_inputs = x.size();
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    for (int in = 0; in < params.num_inputs; in++)
    {
        int32_t *data = static_cast<int32_t *>(x[in]->data);
        vector<int64_t> shape = vector<int64_t>(x[in]->shape, x[in]->shape + x[in]->ndim);
        tshape[in] = shape;
        int in_size = 1;
        for (int i = 0; i < x[in]->ndim; i++)
        {
            in_size *= shape[i];
        }
        vector<int32_t> x_data(in_size);
        for (int i = 0; i < in_size; i++)
        {
            x_data[i] = data[i];
        }
        tdata[in] = x_data;
    }

    string attr_str = "{\"axis\":\"" + to_string(axis) + "\"}";
    NodeAttrs attr;
    LoadOp("concatenate", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *transpose(DLTensor *x, TShape axes)
{
    CVMOpParam params;
    params.func_name = "transpose";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;

    string attr_str = "{\"axes\":\"[";
    for (int i = 0; i < axes.ndim() - 1; i++)
    {
        attr_str += to_string(axes[i]) + ", ";
    }
    attr_str += to_string(axes[axes.ndim() - 1]) + "]\"}";
    NodeAttrs attr;
    LoadOp("transpose", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}

DLTensor *take(DLTensor *x, DLTensor *y, int axis)
{
    CVMOpParam params;
    params.func_name = "take";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    int32_t *xdata = static_cast<int32_t *>(x->data);
    vector<int64_t> xshape = vector<int64_t>(x->shape, x->shape + x->ndim);
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = xdata[i];
    }
    tdata[0] = x_data;
    // y
    int32_t *ydata = static_cast<int32_t *>(y->data);
    vector<int64_t> yshape = vector<int64_t>(y->shape, y->shape + y->ndim);
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < y->ndim; i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = ydata[i];
    }
    tdata[1] = y_data;

    string attr_str = "{\"axis\":\"" + to_string(axis) + "\"}";
    NodeAttrs attr;
    LoadOp("take", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return res.operator->();
}


// int32_t *
data_and_shape conv2d(int32_t *data, vector<int64_t> dshape, int32_t *weight, vector<int64_t> wshape, TShape padding, TShape strides, TShape dilation, int groups, vector<int64_t> bshape, int32_t *bias = nullptr)
{
    CVMOpParam params;
    params.func_name = "conv2d";
    if (bias)
        params.num_inputs = 3;
    else
        params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // data
    tshape[0] = dshape;
    int dsize = 1;
    for (int i = 0; i < dshape.size(); i++)
    {
        dsize *= dshape[i];
    }
    vector<int32_t> d_data(dsize);
    for (int i = 0; i < dsize; i++)
    {
        d_data[i] = data[i];
    }
    tdata[0] = d_data;
    // weight
    tshape[1] = wshape;
    int wsize = 1;
    for (int i = 0; i < wshape.size(); i++)
    {
        wsize *= wshape[i];
    }
    vector<int32_t> w_data(wsize);
    for (int i = 0; i < wsize; i++)
    {
        w_data[i] = weight[i];
    }
    tdata[1] = w_data;
    // bias
    if (bias != nullptr)
    {
        tshape[2] = bshape;
        int bsize = bshape[0];
        vector<int32_t> b_data(bsize);
        for (int i = 0; i < bsize; i++)
        {
            b_data[i] = bias[i];
        }
        tdata[2] = b_data;
    }

    string use_bias = bias == nullptr ? "False" : "True";
    string attr_str = "{\"channels\":" + to_string(dshape[0]) + "\", \"kernel_size\":\"[" + to_string(dshape[2]) + ", " + to_string(dshape[3]) +
                      "], \"strides\":\"[" + to_string(strides[0]) + ", " + to_string(strides[1]) + "]\", \"padding\":\"[" + to_string(padding[0]) +
                      ", " + to_string(padding[1]) + "]\", \"dilation\":\"(" + to_string(dilation[0]) + ", " + to_string(dilation[1]) + ")\", \"groups\":\"" +
                      to_string(groups) + "\", \"layout\":\"NCHW\", \"kernel_layout\":\"OIHW\", \"use_bias\":\"" + use_bias + "\"}";
    NodeAttrs attr;
    LoadOp("conv2d", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape dense(int32_t *data, vector<int64_t> dshape, int32_t *weight, vector<int64_t> wshape, vector<int64_t> bshape, int32_t *bias = nullptr)
{
    CVMOpParam params;
    params.func_name = "dense";
    if (bias)
        params.num_inputs = 3;
    else
        params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // data
    tshape[0] = dshape;
    int dsize = 1;
    for (int i = 0; i < dshape.size(); i++)
    {
        dsize *= dshape[i];
    }
    vector<int32_t> d_data(dsize);
    for (int i = 0; i < dsize; i++)
    {
        d_data[i] = data[i];
    }
    tdata[0] = d_data;
    // weight
    tshape[1] = wshape;
    int wsize = 1;
    for (int i = 0; i < wshape.size(); i++)
    {
        wsize *= wshape[i];
    }
    vector<int32_t> w_data(wsize);
    for (int i = 0; i < wsize; i++)
    {
        w_data[i] = weight[i];
    }
    tdata[1] = w_data;
    // bias
    if (bias != nullptr)
    {
        tshape[2] = bshape;
        int bsize = bshape[0];
        vector<int32_t> b_data(bsize);
        for (int i = 0; i < bsize; i++)
        {
            b_data[i] = bias[i];
        }
        tdata[2] = b_data;
    }

    string use_bias = bias == nullptr ? "False" : "True";
    string attr_str = "\"units\":\"" + to_string(wshape[0]) + "\", \"use_bias\":\"" + use_bias + "\"}";
    NodeAttrs attr;
    LoadOp("dense", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape max_pool2d(int32_t *x, vector<int64_t> xshape, TShape pool_size, TShape strides, TShape padding, bool ceil_mode)
{
    CVMOpParam params;
    params.func_name = "max_pool2d";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;

    string ceil = ceil_mode == true ? "True" : "False";
    string attr_str = "{\"pool_size\":\"" + to_string(pool_size[0]) + ", " + to_string(pool_size[1]) +
                      "]\", \"strides\":\"[" + to_string(padding[0]) + ", " + to_string(padding[1]) + "]\", \"ceil_mode\":\"" +
                      ceil + "\"}";
    NodeAttrs attr;
    LoadOp("max_pool2d", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape upsampling(int32_t *x, vector<int64_t> xshape, int scale)
{
    CVMOpParam params;
    params.func_name = "upsampling";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;

    string attr_str = "{\"scale\":" + to_string(scale) + "\"}";
    NodeAttrs attr;
    LoadOp("upsampling", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

vector<data_and_shape> get_valid_counts(int32_t *x, vector<int64_t> xshape, int score_threshold)
{
    CVMOpParam params;
    params.func_name = "get_valid_counts";
    params.num_inputs = 1;
    params.num_outputs = 2;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = xshape[0] * xshape[1] * xshape[2];
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;

    string attr_str = "{\"score_threshold\":\"" + to_string(score_threshold) + "\"}";
    NodeAttrs attr;
    LoadOp("get_valid_counts", attr);
    LoadOpAttr(attr_str, attr);
    vector<NDArray> res = result(tshape, tdata, params, attr);
    vector<data_and_shape> out(res.size());
    for (int i = 0; i < res.size(); i++)
    {
        out[i] = data_and_shape{(int32_t *)res[i]->data, vector<int64_t>(res[i]->shape, res[i]->shape + res[i]->ndim)};
    }
    return out;
}

data_and_shape sum(int32_t *x, vector<int64_t> xshape, TShape axis, bool keepdims, bool exclude)
{
    CVMOpParam params;
    params.func_name = "sum";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;

    string kdims = keepdims == true ? "True" : "False";
    string exc = exclude == true ? "True" : "False";
    string attr_str = "{\"axis\":\"[" + to_string(axis[0]) + ", " + to_string(axis[1]) + ", " +
                      to_string(axis[2]) + "]\", \"keepdims\":\"" + kdims + "\", \"exclude\":\"" + exc + "\"}";
    NodeAttrs attr;
    LoadOp("sum", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape max(int32_t *x, vector<int64_t> xshape, TShape axis, bool keepdims, bool exclude)
{
    CVMOpParam params;
    params.func_name = "max";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;

    string kdims = keepdims == true ? "True" : "False";
    string exc = exclude == true ? "True" : "False";
    string attr_str = "{\"axis\":\"[" + to_string(axis[0]) + ", " + to_string(axis[1]) + ", " +
                      to_string(axis[2]) + "]\", \"keepdims\":\"" + kdims + "\", \"exclude\":\"" + exc + "\"}";
    NodeAttrs attr;
    LoadOp("max", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape broadcast_add(int32_t *x, vector<int64_t> xshape, int32_t *y, vector<int64_t> yshape)
{
    CVMOpParam params;
    params.func_name = "broadcast_add";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;
    // y
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < yshape.size(); i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = y[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_add", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape broadcast_div(int32_t *x, vector<int64_t> xshape, int32_t *y, vector<int64_t> yshape)
{
    CVMOpParam params;
    params.func_name = "broadcast_div";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;
    // y
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < yshape.size(); i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = y[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_div", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape broadcast_greater(int32_t *x, vector<int64_t> xshape, int32_t *y, vector<int64_t> yshape)
{
    CVMOpParam params;
    params.func_name = "broadcast_greater";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;
    // y
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < yshape.size(); i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = y[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_greater", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape broadcast_max(int32_t *x, vector<int64_t> xshape, int32_t *y, vector<int64_t> yshape)
{
    CVMOpParam params;
    params.func_name = "broadcast_max";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;
    // y
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < yshape.size(); i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = y[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_max", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape broadcast_mul(int32_t *x, vector<int64_t> xshape, int32_t *y, vector<int64_t> yshape)
{
    CVMOpParam params;
    params.func_name = "broadcast_mul";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;
    // y
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < yshape.size(); i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = y[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_mul", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape broadcast_sub(int32_t *x, vector<int64_t> xshape, int32_t *y, vector<int64_t> yshape)
{
    CVMOpParam params;
    params.func_name = "broadcast_sub";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;
    // y
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < yshape.size(); i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = y[i];
    }
    tdata[1] = y_data;

    string attr_str = "{}";
    NodeAttrs attr;
    LoadOp("broadcast_sub", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape slice_like(int32_t *x, vector<int64_t> xshape, int32_t *y, vector<int64_t> yshape, TShape axis)
{
    CVMOpParam params;
    params.func_name = "slice_like";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;
    // y
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < yshape.size(); i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = y[i];
    }
    tdata[1] = y_data;

    string attr_str = "{\"axis\":\"[" + to_string(axis[0]) + ", " + to_string(axis[1]) + ", " + to_string(axis[2]) + ", " + to_string(axis[3]) + "]\"";
    NodeAttrs attr;
    LoadOp("slice_like", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape tile(int32_t *x, vector<int64_t> xshape, int32_t *y, vector<int64_t> yshape, TShape reps)
{
    CVMOpParam params;
    params.func_name = "tile";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;
    // y
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < yshape.size(); i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = y[i];
    }
    tdata[1] = y_data;

    string attr_str = "{\"reps\":\"[";
    for (int i = 0; i < reps.ndim() - 1; i++)
    {
        attr_str += to_string(reps[i]) + ", ";
    }
    attr_str += to_string(reps[reps.ndim() - 1]) + "]\"}";
    NodeAttrs attr;
    LoadOp("tile", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape repeat(int32_t *x, vector<int64_t> xshape, int repeats, int axis)
{
    CVMOpParam params;
    params.func_name = "repeat";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;

    string attr_str = "{\"repeats\":\"" + to_string(repeats) + "\", \"axis\":\"" + to_string(axis) + "\"}";
    NodeAttrs attr;
    LoadOp("repeat", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape strided_slice(int32_t *x, vector<int64_t> xshape, TShape begin, TShape end, TShape stride)
{
    CVMOpParam params;
    params.func_name = "strided_slice";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;

    string attr_str = "{\"begin\":\"[";
    for (int i = 0; i < begin.ndim() - 1; i++)
        attr_str += to_string(begin[i]) + ", ";
    attr_str += to_string(begin[begin.ndim() - 1]) + "]\", \"end\":\"[";
    for (int i = 0; i < end.ndim() - 1; i++)
        attr_str += to_string(end[i]) + ", ";
    attr_str += to_string(end[end.ndim() - 1]) + "]\", \"stride\":\"[";
    for (int i = 0; i < stride.ndim() - 1; i++)
        attr_str += to_string(stride[i]) + ", ";
    attr_str += to_string(stride[stride.ndim() - 1]) + "]\"}";
    NodeAttrs attr;
    LoadOp("strided_slice", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape concatenate(vector<int32_t *> x, vector<vector<int64_t>> xshape, int axis)
{
    CVMOpParam params;
    params.func_name = "concatenate";
    params.num_inputs = x.size();
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    for (int in = 0; in < params.num_inputs; in++)
    {
        tshape[in] = xshape[in];
        int in_size = 1;
        for (int i = 0; i < xshape[in].size(); i++)
        {
            in_size *= xshape[in][i];
        }
        vector<int32_t> x_data(in_size);
        for (int i = 0; i < in_size; i++)
        {
            x_data[i] = x[in][i];
        }
        tdata[in] = x_data;
    }

    string attr_str = "{\"axis\":\"" + to_string(axis) + "\"}";
    NodeAttrs attr;
    LoadOp("concatenate", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape transpose(int32_t *x, vector<int64_t> xshape, TShape axes)
{
    CVMOpParam params;
    params.func_name = "transpose";
    params.num_inputs = 1;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;

    string attr_str = "{\"axes\":\"[";
    for (int i = 0; i < axes.ndim() - 1; i++)
    {
        attr_str += to_string(axes[i]) + ", ";
    }
    attr_str += to_string(axes[axes.ndim() - 1]) + "]\"}";
    NodeAttrs attr;
    LoadOp("transpose", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

data_and_shape take(int32_t *x, vector<int64_t> xshape, int32_t *y, vector<int64_t> yshape, int axis)
{
    CVMOpParam params;
    params.func_name = "take";
    params.num_inputs = 2;
    params.num_outputs = 1;
    params.flatten_data = false;

    vector<vector<int64_t>> tshape(params.num_inputs + params.num_outputs);
    vector<vector<int32_t>> tdata(params.num_inputs + params.num_outputs);
    // x
    tshape[0] = xshape;
    int xsize = 1;
    for (int i = 0; i < xshape.size(); i++)
    {
        xsize *= xshape[i];
    }
    vector<int32_t> x_data(xsize);
    for (int i = 0; i < xsize; i++)
    {
        x_data[i] = x[i];
    }
    tdata[0] = x_data;
    // y
    tshape[1] = yshape;
    int ysize = 1;
    for (int i = 0; i < yshape.size(); i++)
    {
        ysize *= yshape[i];
    }
    vector<int32_t> y_data(ysize);
    for (int i = 0; i < ysize; i++)
    {
        y_data[i] = y[i];
    }
    tdata[1] = y_data;

    string attr_str = "{\"axis\":\"" + to_string(axis) + "\"}";
    NodeAttrs attr;
    LoadOp("take", attr);
    LoadOpAttr(attr_str, attr);
    NDArray res = result(tshape, tdata, params, attr)[0];
    return data_and_shape{(int32_t *)res->data, vector<int64_t>(res->shape, res->shape + res->ndim)};
}

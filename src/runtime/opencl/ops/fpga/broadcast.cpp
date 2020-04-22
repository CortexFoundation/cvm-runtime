int broadcast_eval(int type, int a, int b, int &c)
{
    if (type == 0) //broadcast_add
        c = a + b;
    else if (type == 1) //broadcast_sub
        c = a - b;
    else if (type == 2) //broadcast_mul
        c = a * b;
    else if (type == 3) //broadcast_max
        c = a > b ? a : b;
    else if (type == 4) //broadcast_div
        c = b == 0 ? 0 : a / b;
    else if (type == 5) //broadcast_greater
        c = a > b;
}

int broadcast_i_index(int *oshape, int o_index, int *ishape, int idim, int odim)
{
    if (idim == 1 && ishape[0] == 1)
        return 0;
    int index = 0;
    int allIndex = 1;
    for (int i = 0; i < idim; i++)
    {
        int idx = idim - 1 - i;
        int ovar = o_index % oshape[idx + odim - idim];
        if (ovar < ishape[idx])
        {
            index += allIndex * ovar;
        }
        allIndex = allIndex * ishape[idx];
        o_index /= oshape[idx + odim - idim];
    }
    return index;
}

int getSize(int *shape, int ndim)
{
    int size = 1;
    for (int i = 0; i < ndim; i++)
    {
        size *= shape[i];
    }
    return size;
}

extern "C"
{
    void broadcast(
        int *a, int *b, int *c,
        int *ashape, int adim,
        int *bshape, int bdim,
        int *cshape, int cdim,
        int type)
    {
#pragma HLS INTERFACE m_axi port = a offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = b offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = c offset = slave bundle = gmem

#pragma HLS INTERFACE m_axi port = ashape offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = bshape offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = cshape offset = slave bundle = gmem

#pragma HLS INTERFACE s_axilite port = a bundle = control
#pragma HLS INTERFACE s_axilite port = b bundle = control
#pragma HLS INTERFACE s_axilite port = c bundle = control
#pragma HLS INTERFACE s_axilite port = ashape bundle = control
#pragma HLS INTERFACE s_axilite port = bshape bundle = control
#pragma HLS INTERFACE s_axilite port = cshape bundle = control
#pragma HLS INTERFACE s_axilite port = adim bundle = control
#pragma HLS INTERFACE s_axilite port = bdim bundle = control
#pragma HLS INTERFACE s_axilite port = cdim bundle = control
#pragma HLS INTERFACE s_axilite port = type bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

        int a_size = getSize(ashape, adim);
        int b_size = getSize(bshape, bdim);
        int c_size = getSize(cshape, cdim);

        if (a_size == 1)
            for (int i = 0; i < c_size; i++)
                broadcast_eval(type, a[0], b[i], c[i]);
        else if (b_size == 1)
            for (int i = 0; i < c_size; i++)
                broadcast_eval(type, a[i], b[0], c[i]);
        else
        {
            for (int i = 0; i < c_size; i++)
            {
#pragma HLS PIPELINE II = 1
                int a_index = broadcast_i_index(cshape, i, ashape, adim, cdim);
                int b_index = broadcast_i_index(cshape, i, bshape, bdim, cdim);
                broadcast_eval(type, a[a_index], b[b_index], c[i]);
            }
        }
    }
}

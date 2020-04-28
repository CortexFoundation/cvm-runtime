const int MAX_DIM = 6;

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

int broadcast_i_index(int oshape[], int o_index, int ishape[], int idim, int odim)
{
    int index = 0;
    int allIndex = 1;
    for (int i = 0; i < idim; i++)
    {
        int idx = idim - 1 - i;
        int ovar = o_index % oshape[idx + odim - idim + MAX_DIM - odim];
        if (ovar < ishape[idx + MAX_DIM - idim])
        {
            index += allIndex * ovar;
        }
        allIndex = allIndex * ishape[idx + MAX_DIM - idim];
        o_index /= oshape[idx + odim - idim + MAX_DIM - odim];
    }
    return index;
}

extern "C"
{
    void broadcast(
        int *a, int *b, int *c,
        int asize, int bsize, int csize,
        int adim, int bdim, int cdim,
        int ashape0, int ashape1, int ashape2, int ashape3, int ashape4, int ashape5,
        int bshape0, int bshape1, int bshape2, int bshape3, int bshape4, int bshape5,
        int cshape0, int cshape1, int cshape2, int cshape3, int cshape4, int cshape5,
        int type)
    {
#pragma HLS INTERFACE m_axi port = a offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = b offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = c offset = slave bundle = gmem

#pragma HLS INTERFACE s_axilite port = a bundle = control
#pragma HLS INTERFACE s_axilite port = b bundle = control
#pragma HLS INTERFACE s_axilite port = c bundle = control
#pragma HLS INTERFACE s_axilite port = adim bundle = control
#pragma HLS INTERFACE s_axilite port = bdim bundle = control
#pragma HLS INTERFACE s_axilite port = cdim bundle = control
#pragma HLS INTERFACE s_axilite port = type bundle = control

#pragma HLS INTERFACE s_axilite port = asize bundle = control
#pragma HLS INTERFACE s_axilite port = bsize bundle = control
#pragma HLS INTERFACE s_axilite port = csize bundle = control

#pragma HLS INTERFACE s_axilite port = ashape0 bundle = control
#pragma HLS INTERFACE s_axilite port = ashape1 bundle = control
#pragma HLS INTERFACE s_axilite port = ashape2 bundle = control
#pragma HLS INTERFACE s_axilite port = ashape3 bundle = control
#pragma HLS INTERFACE s_axilite port = ashape4 bundle = control
#pragma HLS INTERFACE s_axilite port = ashape5 bundle = control

#pragma HLS INTERFACE s_axilite port = bshape0 bundle = control
#pragma HLS INTERFACE s_axilite port = bshape1 bundle = control
#pragma HLS INTERFACE s_axilite port = bshape2 bundle = control
#pragma HLS INTERFACE s_axilite port = bshape3 bundle = control
#pragma HLS INTERFACE s_axilite port = bshape4 bundle = control
#pragma HLS INTERFACE s_axilite port = bshape5 bundle = control

#pragma HLS INTERFACE s_axilite port = cshape0 bundle = control
#pragma HLS INTERFACE s_axilite port = cshape1 bundle = control
#pragma HLS INTERFACE s_axilite port = cshape2 bundle = control
#pragma HLS INTERFACE s_axilite port = cshape3 bundle = control
#pragma HLS INTERFACE s_axilite port = cshape4 bundle = control
#pragma HLS INTERFACE s_axilite port = cshape5 bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

        int ashape[MAX_DIM] = {ashape0, ashape1, ashape2, ashape3, ashape4, ashape5};
        int bshape[MAX_DIM] = {bshape0, bshape1, bshape2, bshape3, bshape4, bshape5};
        int cshape[MAX_DIM] = {cshape0, cshape1, cshape2, cshape3, cshape4, cshape5};

        if (asize == 1)
            for (int i = 0; i < csize; i++)
                broadcast_eval(type, a[0], b[i], c[i]);
        else if (bsize == 1)
            for (int i = 0; i < csize; i++)
                broadcast_eval(type, a[i], b[0], c[i]);
        else
        {
            for (int i = 0; i < csize; i++)
            {
#pragma HLS PIPELINE II = 1
                int a_index = broadcast_i_index(cshape, i, ashape, adim, cdim);
                int b_index = broadcast_i_index(cshape, i, bshape, bdim, cdim);
                broadcast_eval(type, a[a_index], b[b_index], c[i]);
            }
        }
    }
}

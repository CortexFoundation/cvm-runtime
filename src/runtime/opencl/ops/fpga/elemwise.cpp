extern "C"
{

    void elemwise(int *a, int *b, int *c, int a_size, int type)
    {
#pragma HLS INTERFACE m_axi port = a offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = b offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = c offset = slave bundle = gmem

#pragma HLS INTERFACE s_axilite port = a bundle = control
#pragma HLS INTERFACE s_axilite port = b bundle = control
#pragma HLS INTERFACE s_axilite port = c bundle = control

#pragma HLS INTERFACE s_axilite port = a_size bundle = control
#pragma HLS INTERFACE s_axilite port = type bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

        if (type == 0)
        {
            for (int i = 0; i < a_size; i++)
            {
#pragma HLS PIPELINE II = 1
                c[i] = a[i] + b[i];
            }
        }
        else if (type == 1)
        {
            for (int i = 0; i < a_size; i++)
            {
#pragma HLS PIPELINE II = 1
                c[i] = a[i] - b[i];
            }
        }
    }
}

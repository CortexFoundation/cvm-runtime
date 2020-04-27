extern "C"
{

    void clip(int *a, int *b, int a_size, int max, int min)
    {
#pragma HLS INTERFACE m_axi port = a offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = b offset = slave bundle = gmem

#pragma HLS INTERFACE s_axilite port = a bundle = control
#pragma HLS INTERFACE s_axilite port = b bundle = control

#pragma HLS INTERFACE s_axilite port = a_size bundle = control
#pragma HLS INTERFACE s_axilite port = max bundle = control
#pragma HLS INTERFACE s_axilite port = min bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control
        for (int i = 0; i < a_size; i++)
        {
#pragma HLS PIPELINE II = 1
            int musk = max > a[i] ? a[i] : max;
            b[i] = musk > min ? musk : min;
        }
    }
}

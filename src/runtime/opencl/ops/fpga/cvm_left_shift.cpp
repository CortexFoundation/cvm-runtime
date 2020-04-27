extern "C"
{

    void cvm_left_shift(int *a, int *c, int a_size, int shift_bit, int precision)
    {
#pragma HLS INTERFACE m_axi port = a offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = c offset = slave bundle = gmem

#pragma HLS INTERFACE s_axilite port = a bundle = control
#pragma HLS INTERFACE s_axilite port = c bundle = control

#pragma HLS INTERFACE s_axilite port = a_size bundle = control
#pragma HLS INTERFACE s_axilite port = shift_bit bundle = control
#pragma HLS INTERFACE s_axilite port = precision bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

        int min = -(((int)1 << (precision - 1)) - 1);
        int max = -min;

        for (int i = 0; i < a_size; i++)
        {
#pragma HLS PIPELINE II = 1
            int shift_a = a[i] << shift_bit;
            int musk = max > shift_a ? shift_a : max;
            c[i] = musk > min ? musk : min;
        }
    }
}

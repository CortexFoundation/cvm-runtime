
#define BUFFER_SIZE 64


extern "C"
{
    void broadcast_mul(const int *a, const int b, int *c, const int n)
    {
#pragma HLS INTERFACE m_axi port = a offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = c offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = a bundle = control
#pragma HLS INTERFACE s_axilite port = b bundle = control
#pragma HLS INTERFACE s_axilite port = c bundle = control
#pragma HLS INTERFACE s_axilite port = n bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

        unsigned int v1_buffer[BUFFER_SIZE];
        unsigned int vout_buffer[BUFFER_SIZE];

        for (int i = 0; i < n; i += BUFFER_SIZE)
        {
            int chunk_size = BUFFER_SIZE;
            if ((i + BUFFER_SIZE) > n)
                chunk_size = n - i;
        read1:
            for (int j = 0; j < chunk_size; j++)
            {
				#pragma HLS PIPELINE II = 1
                v1_buffer[j] = a[i + j];
            }
        broadcast_mul:
            for (int j = 0; j < chunk_size; j++)
            {
				#pragma HLS PIPELINE II = 1
                vout_buffer[j] = v1_buffer[j] * b;
            }
        write:
            for (int j = 0; j < chunk_size; j++)
            {
				#pragma HLS PIPELINE II = 1
                c[i + j] = vout_buffer[j];
            }
        }
    }
}

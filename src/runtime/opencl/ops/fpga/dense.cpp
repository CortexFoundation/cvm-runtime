
extern "C"
{
    void dense(const int *a, const int *b, int *c, const int M, const int N, const int K)
    {
#pragma HLS INTERFACE m_axi port = a offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = b offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = c offset = slave bundle = gmem

#pragma HLS INTERFACE s_axilite port = a bundle = control
#pragma HLS INTERFACE s_axilite port = b bundle = control
#pragma HLS INTERFACE s_axilite port = c bundle = control

#pragma HLS INTERFACE s_axilite port = M bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = K bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control
        int BUFF_SIZE = 64;

        for (int i = 0; i < M; i += BUFF_SIZE)
        {
            int chunk_size_i = BUFF_SIZE;
            if (i + BUFF_SIZE > M)
                chunk_size_i = M - i;
            for (int j = 0; j < N; j += BUFF_SIZE)
            {
                int chunk_size_j = BUFF_SIZE;
                if (j + BUFF_SIZE > N)
                    chunk_size_j = N - j;

            dense:
                for (int q = 0; q < chunk_size_i; q++)
                {
                    for (int p = 0; p < chunk_size_j; p++)
                    {
                        for (int r = 0; r < K; r++)
                        {
                            #pragma HLS PIPELINE II = 1
                            c[(i + q) * N + j + p] += a[(i + q) * K + r] * b[(j + p) * K + r];
                        }
                    }
                }
            }
        }
    }
}

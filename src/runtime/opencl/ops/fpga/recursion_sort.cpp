extern "C"
{

    void recursion_sort(int *a, int *c, const int M, const int N, const int K)
    {
#pragma HLS INTERFACE m_axi port = a offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = c offset = slave bundle = gmem

#pragma HLS INTERFACE s_axilite port = a bundle = control
#pragma HLS INTERFACE s_axilite port = c bundle = control

#pragma HLS INTERFACE s_axilite port = M bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = K bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control
        for (int i = 2; i < M * 2; i *= 2)
        {
            for (int j = 0; j < (M + i - 1) / i; j++)
            {
#pragma HLS PIPELINE II = 1
                int left = i * j;
                int mid = left + i / 2 - 1 >= M ? (M - 1) : (left + i / 2 - 1);
                int right = i * (j + 1) - 1 >= M ? (M - 1) : (i * (j + 1) - 1);

                int k = left * N, l = left, r = mid + 1;
                while (l <= mid && r <= right)
                    if (a[l * N + K] <= a[r * N + K])
                    {
                        for (int m = l * N; m < l * N + N; m++)
                            c[k++] = a[m];
                        l++;
                    }
                    else
                    {
                        for (int m = r * N; m < r * N + N; m++)
                            c[k++] = a[m];
                        r++;
                    }

                while (l <= mid)
                {
                    for (int m = l * N; m < l * N + N; m++)
                        c[k++] = a[m];
                    l++;
                }
                while (r <= right)
                {
                    for (int m = r * N; m < r * N + N; m++)
                        c[k++] = a[m];
                    r++;
                }
                for (int m = left * N; m < (right + 1) * N; m++)
                    a[m] = c[m];
            }
        }
    }
}
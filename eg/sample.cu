extern "C" __global__ void kernel (float *a, float *b, float *c, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    c[col + row * n] = 0;
    if(col < n && row < n) {
        float tmp = 0.0f;
        for (int i = 0; i < n; i++) {
            tmp = tmp + a[row * n + i] * a[col + n * i];
        }
        c[col + row * n] = tmp;
    }
}

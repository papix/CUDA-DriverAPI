extern "C" __global__ void kernel_sum(float *a, float *b, float *c, int *i)
{
    int tid = blockIdx.x;
    if (tid < *i) {
        c[tid] = a[tid] + b[tid];
    }
}

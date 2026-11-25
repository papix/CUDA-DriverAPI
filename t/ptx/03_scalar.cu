extern "C" __global__ void kernel_axpy(float a, const float *x, float *y, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        y[tid] = a * x[tid] + y[tid];
    }
}

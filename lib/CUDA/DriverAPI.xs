#ifdef __cplusplus
extern "C" {
#endif

#define PERL_NO_GET_CONTEXT /* we want efficiency */
#include <EXTERN.h>
#include <perl.h>
#include <XSUB.h>

#ifdef __cplusplus
} /* extern "C" */
#endif

#define NEED_newSVpvn_flags
#include "ppport.h"

MODULE = CUDA::DriverAPI    PACKAGE = CUDA::DriverAPI

PROTOTYPES: DISABLE

#include <cuda.h>

SV *
_init()
CODE:
{
    cuInit(0);

    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUcontext ctx;
    cuCtxCreate(&ctx, 0, dev);

    RETVAL = newSViv(PTR2IV(ctx));
}
OUTPUT:
    RETVAL

SV *
_malloc(SV * ctx, SV * data_SV)
CODE:
{
    CUcontext cuda_context = INT2PTR(CUcontext, SvIV(ctx));
    CUdeviceptr dev_ptr;

    size_t data_len = 0;
    data_len = (size_t)SvCUR(data_SV);

    CUresult err = cuMemAlloc(&dev_ptr, data_len);
    if (err != CUDA_SUCCESS) {
        croak("Error at memory allocation (%04d)", err);
    }
    RETVAL = newSViv(PTR2IV(dev_ptr));
}
OUTPUT:
    RETVAL

void
_transfer_h2d(SV * ctx, SV * src_SV, SV * dst_SV)
CODE:
{
    CUcontext cuda_context = INT2PTR(CUcontext, SvIV(ctx));
    void * src_ptr = sv_2pvbyte_nolen(src_SV);
    CUdeviceptr dst_ptr = INT2PTR(CUdeviceptr, SvIV(dst_SV));
    size_t data_len = (size_t)SvCUR(src_SV);

    CUresult err = cuMemcpyHtoD(dst_ptr, src_ptr, data_len);
    if (err != CUDA_SUCCESS) {
        croak("Error at transfer host to device (%04d)", err);
    }
}

void
_transfer_d2h(SV * ctx, SV * src_SV, SV * dst_SV)
CODE:
{
    CUcontext cuda_context = INT2PTR(CUcontext, SvIV(ctx));
    CUdeviceptr src_ptr = INT2PTR(CUdeviceptr, SvIV(src_SV));
    void * dst_ptr = sv_2pvbyte_nolen(dst_SV);
    size_t data_len = (size_t)SvCUR(dst_SV);

    CUresult err = cuMemcpyDtoH(dst_ptr, src_ptr, data_len);
    if (err != CUDA_SUCCESS) {
        croak("Error at transfer device to host (%04d)", err);
    }
}

void
_run(SV * ctx, ptx_path, function, AV * args, AV * config)
    char* ptx_path
    char* function
CODE:
{
    CUcontext cuda_context = INT2PTR(CUcontext, SvIV(ctx));
    CUresult err;

    int i;
    int args_size = av_len(args);
    void * args_ptr[args_size];
    void * kernel_param[args_size];

    for (i = 0; i <= args_size; i++) {
        SV * arg = *av_fetch(args, i, FALSE);
        args_ptr[i] = (void *) INT2PTR(CUdeviceptr, SvIV(arg));
        kernel_param[i] = &args_ptr[i];
    }

    int cnf[] = {
        1, 1, 1, 1, 1, 1, 0,
    };

    int config_size = av_len(config);
    for (i = 0; i <= config_size; i++) {
        cnf[i] = SvIV((SV *)*av_fetch(config, i, FALSE));
    }

    CUmodule module;
    err = cuModuleLoad(&module, ptx_path);
    if (err != CUDA_SUCCESS) {
        croak("Error at ModuleLoad (%04d)", err);
    }

    CUfunction func;
    err = cuModuleGetFunction(&func, module, function);
    if (err != CUDA_SUCCESS) {
        croak("Error at ModuleGetFunction (%04d)", err);
    }

    err = cuLaunchKernel(func, cnf[0], cnf[1], cnf[2], cnf[3], cnf[4], cnf[5], cnf[6], 0, kernel_param, 0);
    if (err != CUDA_SUCCESS) {
        croak("Error at LaunchKernel (%04d)", err);
    }

    cuCtxSynchronize();
}

void
_free(SV * ctx, SV * data_SV)
CODE:
{
    CUcontext cuda_context = INT2PTR(CUcontext, SvIV(ctx));
    CUresult err = cuMemFree(INT2PTR(CUdeviceptr, SvIV(data_SV)));
    if (err != CUDA_SUCCESS) {
        croak("Error at MemFree (%04d)", err);
    }
}

void
_destroy(SV * ctx)
CODE:
{
    CUcontext cuda_context = INT2PTR(CUcontext, SvIV(ctx));
    CUresult err = cuCtxDetach(cuda_context);
    if (err != CUDA_SUCCESS) {
        croak("Error at CtxDetach (%04d)", err);
    }
}

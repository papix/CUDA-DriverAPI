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

#include "ppport.h"

#include <dlfcn.h>
#include <unistd.h>
#include <float.h>
#include <math.h>

#include <cuda.h>
#include <nvrtc.h>

#if CUDA_VERSION < 12000
#error "CUDA::DriverAPI requires CUDA 12.0 or later (cuLibrary* API)"
#endif

/* ------------------------------------------------------------------------
 * libcuda comes with the NVIDIA driver, so it is not linked but loaded at
 * run time by load_driver(): the module then loads, and its tests skip, on
 * machines with the toolkit but no driver. Every driver call goes through
 * CU(name), a pointer with the exact type of the cuda.h prototype (which
 * may be a _v2 variant), resolved with cuGetProcAddress.
 * ------------------------------------------------------------------------ */

#define CU_FUNCTIONS(X)                                                         \
    X(cuInit) X(cuDeviceGet) X(cuDeviceGetAttribute)                            \
    X(cuDevicePrimaryCtxRetain) X(cuDevicePrimaryCtxRelease)                    \
    X(cuCtxSetCurrent) X(cuCtxSynchronize) X(cuGetErrorName) X(cuGetErrorString) \
    X(cuMemAlloc) X(cuMemFree) X(cuMemcpyHtoD) X(cuMemcpyDtoH) X(cuMemsetD8Async) \
    X(cuLibraryLoadFromFile) X(cuLibraryLoadData) X(cuLibraryUnload)             \
    X(cuLibraryGetKernel) X(cuKernelGetFunction) X(cuLaunchKernel)

#define CU_DECLARE(name) static __typeof__(&name) p_##name;
CU_FUNCTIONS(CU_DECLARE)
#undef CU_DECLARE

#define CU(name) p_##name

static const struct {
    const char *name;
    void **slot;
} CU_TABLE[] = {
#define CU_ENTRY(name) { #name, (void **)&p_##name },
    CU_FUNCTIONS(CU_ENTRY)
#undef CU_ENTRY
};

/* Optional, from CUDA 12.4 drivers on: lets run() check kernel arguments.
 * Declared by hand so that the module still builds with older headers. */
typedef CUresult (CUDAAPI *func_get_param_info_t)(CUfunction func, size_t index, size_t *offset, size_t *size);
static func_get_param_info_t p_func_get_param_info;

static int driver_loaded = 0;

/* Gives up a partial load, so that a later new() starts over cleanly. */
static void
unload_driver(void *lib)
{
    size_t i;

    for (i = 0; i < sizeof(CU_TABLE) / sizeof(CU_TABLE[0]); i++) {
        *CU_TABLE[i].slot = NULL;
    }
    p_func_get_param_info = NULL;
    dlclose(lib);
}

static void
load_driver(pTHX)
{
    void *lib;
    __typeof__(&cuDriverGetVersion) get_version;
    __typeof__(&cuGetProcAddress) get_proc;
    int driver_version = 0;
    int version;
    CUresult err;
    size_t i;

    if (driver_loaded) {
        return;
    }
    /* WSL keeps the driver's libcuda in /usr/lib/wsl/lib, which the loader
     * normally knows through ld.so.conf; try it directly as a fallback. */
    lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        const char *message = dlerror();
        SV *error = sv_2mortal(newSVpv(message ? message : "unknown dlopen error", 0));

        lib = dlopen("/usr/lib/wsl/lib/libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib) {
            /* Report the WSL attempt too unless the file is simply absent. */
            const char *wsl_error = dlerror();

            if (access("/usr/lib/wsl/lib/libcuda.so.1", F_OK) == 0) {
                croak("CUDA driver not found: %" SVf "; %s", SVfARG(error), wsl_error ? wsl_error : "unknown dlopen error");
            }
            croak("CUDA driver not found: %" SVf, SVfARG(error));
        }
    }
    get_version = (__typeof__(get_version))dlsym(lib, "cuDriverGetVersion");
    get_proc    = (__typeof__(get_proc))dlsym(lib, "cuGetProcAddress_v2");
    if (!get_version || !get_proc) {
        unload_driver(lib);
        croak("CUDA driver too old: libcuda.so.1 lacks cuGetProcAddress_v2 (CUDA 12.0 or later is required)");
    }
    if ((err = get_version(&driver_version)) != CUDA_SUCCESS) {
        unload_driver(lib);
        croak("CUDA driver not usable: cuDriverGetVersion returned CUresult %d"
              " (34 means the toolkit's stub libcuda was loaded instead of a driver)", (int)err);
    }
    if (driver_version < 12000) {
        unload_driver(lib);
        croak("CUDA driver too old: version %d, CUDA 12.0 or later is required", driver_version);
    }

    /* Asking for a version newer than the driver's is an error, so ask for
     * the driver's version when the headers are newer. The entry points used
     * here kept their ABI from CUDA 12.0 on. The legacy-stream variants keep
     * the ordering that run() and fill() rely on (see AGENTS.md). */
    version = driver_version < CUDA_VERSION ? driver_version : CUDA_VERSION;
    for (i = 0; i < sizeof(CU_TABLE) / sizeof(CU_TABLE[0]); i++) {
        CUdriverProcAddressQueryResult status;

        if (get_proc(CU_TABLE[i].name, CU_TABLE[i].slot, version, CU_GET_PROC_ADDRESS_LEGACY_STREAM, &status)
                != CUDA_SUCCESS
            || status != CU_GET_PROC_ADDRESS_SUCCESS || !*CU_TABLE[i].slot) {
            unload_driver(lib);
            croak("CUDA driver (version %d) does not provide %s", driver_version, CU_TABLE[i].name);
        }
    }
    {
        /* Its ABI is the one declared above whatever the headers' version. */
        CUdriverProcAddressQueryResult status;

        if (get_proc("cuFuncGetParamInfo", (void **)&p_func_get_param_info, driver_version,
                     CU_GET_PROC_ADDRESS_LEGACY_STREAM, &status) != CUDA_SUCCESS
            || status != CU_GET_PROC_ADDRESS_SUCCESS) {
            p_func_get_param_info = NULL;
        }
    }
    driver_loaded = 1;
}

/* Number of values accepted by cuLaunchKernel's launch configuration:
 * gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes */
#define LAUNCH_CONFIG_SIZE 7

static void
check_cu(pTHX_ CUresult err, const char *what)
{
    const char *name = NULL;
    const char *desc = NULL;

    if (err == CUDA_SUCCESS) {
        return;
    }
    /* The lookups themselves fail on e.g. the libcuda stub; fall back to the code. */
    if (CU(cuGetErrorName)(err, &name) != CUDA_SUCCESS || CU(cuGetErrorString)(err, &desc) != CUDA_SUCCESS) {
        croak("CUDA error at %s: CUresult %d", what, (int)err);
    }
    croak("CUDA error at %s: %s (%d): %s", what, name, (int)err, desc);
}

#define CHECK_CU(call, what) check_cu(aTHX_ (call), (what))

static void
check_nvrtc(pTHX_ nvrtcResult err, const char *what)
{
    if (err != NVRTC_SUCCESS) {
        croak("NVRTC error at %s: %s (%d)", what, nvrtcGetErrorString(err), (int)err);
    }
}

#define CHECK_NVRTC(call, what) check_nvrtc(aTHX_ (call), (what))

static void
set_buffer_length(pTHX_ SV *sv, size_t len)
{
    SvCUR_set(sv, len);
    *SvEND(sv) = '\0';
}

/* Empty mortal byte string with room for len bytes plus a trailing NUL;
 * mortal so that a croak before it is returned does not leak it. */
static SV *
new_buffer(pTHX_ size_t len)
{
    SV *sv = sv_2mortal(newSV(len ? len : 1));

    SvPOK_only(sv);
    set_buffer_length(aTHX_ sv, 0);
    return sv;
}

/* Every entry point makes the object's context current, so that the calling
 * thread always operates on the right device. */
#define SET_CONTEXT(ctx_sv)                                                              \
    do {                                                                                 \
        load_driver(aTHX);                                                               \
        CHECK_CU(CU(cuCtxSetCurrent)(INT2PTR(CUcontext, SvIV(ctx_sv))), "CtxSetCurrent"); \
    } while (0)

/* Croaks with the same message as the Perl side when a transfer would run past
 * the end of a tracked allocation. */
static void
check_bounds(pTHX_ STRLEN len, UV limit)
{
    if (len > limit) {
        croak("Transfer of %" UVuf " bytes exceeds the allocation of %" UVuf " bytes", (UV)len, limit);
    }
}

/* ------------------------------------------------------------------------
 * Nested array <-> packed element conversion for Buffer->write_array and
 * Buffer->read_array. Arrays are rectangular, row-major, of rank 1..3.
 * ------------------------------------------------------------------------ */

#define MAX_RANK 3

typedef enum {
    ET_I8, ET_U8, ET_I16, ET_U16, ET_I32, ET_U32, ET_I64, ET_U64, ET_F32, ET_F64
} elem_type;

static const struct {
    const char *name;
    elem_type type;
    size_t size;
} ELEM_TYPES[] = {
    { "i8",  ET_I8,  1 }, { "u8",  ET_U8,  1 },
    { "i16", ET_I16, 2 }, { "u16", ET_U16, 2 },
    { "i32", ET_I32, 4 }, { "u32", ET_U32, 4 },
    { "i64", ET_I64, 8 }, { "u64", ET_U64, 8 },
    { "f32", ET_F32, 4 }, { "f64", ET_F64, 8 },
};

static elem_type
lookup_elem_type(pTHX_ const char *name, size_t *size)
{
    size_t i;

    for (i = 0; i < sizeof(ELEM_TYPES) / sizeof(ELEM_TYPES[0]); i++) {
        if (strEQ(name, ELEM_TYPES[i].name)) {
            *size = ELEM_TYPES[i].size;
            return ELEM_TYPES[i].type;
        }
    }
    croak("Unknown array element type: %s", name);
    return ET_F32; /* not reached */
}

/* "[3][1]" for the first depth indices of path, for error messages. */
static const char *
format_path(pTHX_ const size_t *path, int depth)
{
    SV *sv = sv_2mortal(newSVpvs(""));
    int i;

    for (i = 0; i < depth; i++) {
        sv_catpvf(sv, "[%" UVuf "]", (UV)path[i]);
    }
    return SvPV_nolen(sv);
}

/* Reading a tied array element or an element with get-magic runs Perl code,
 * which may shrink, grow or free the arrays being converted; so may a
 * __WARN__ handler called while a string is numified. Whenever such code may
 * have run, *magic is set: from that point on, every row is checked for its
 * length before it is read again. */
static void
check_unchanged(pTHX_ AV *row, size_t n, const size_t *path, int depth)
{
    if ((size_t)av_count(row) != n) {
        croak("Row %s changed while it was being converted", format_path(aTHX_ path, depth));
    }
}

/* Element i of a row of n elements with its get-magic already applied, so
 * callers use the _nomg conversions. A magical element is kept alive until
 * the end of the statement in case its magic drops the other references. */
static SV *
fetch_elem(pTHX_ AV *row, size_t i, size_t n, const size_t *path, int depth, int *magic)
{
    SV *sv;

    /* Magic on an element of a nested row may have changed this row too. */
    if (*magic) {
        check_unchanged(aTHX_ row, n, path, depth);
    }
    if (!SvRMAGICAL(row)) {
        sv = AvARRAY(row)[i];
    }
    else {
        SV **svp = av_fetch(row, (SSize_t)i, FALSE);
        sv     = svp ? *svp : NULL;
        *magic = 1;
        check_unchanged(aTHX_ row, n, path, depth);
    }
    if (sv && SvGMAGICAL(sv)) {
        sv_2mortal(SvREFCNT_inc_simple_NN(sv));
        mg_get(sv);
        *magic = 1;
        check_unchanged(aTHX_ row, n, path, depth);
    }
    return sv;
}

/* The array an element refers to, kept alive until the caller's scope ends. */
static AV *
elem_as_av(pTHX_ SV *sv, const size_t *path, int depth)
{
    AV *av;

    if (!sv || !SvROK(sv) || SvTYPE(SvRV(sv)) != SVt_PVAV) {
        croak("Element %s is not an array reference", format_path(aTHX_ path, depth));
    }
    av = (AV *)SvRV(sv);
    SvREFCNT_inc_simple_void_NN(av);
    SAVEFREESV(av);
    return av;
}

/* Casting a finite double outside float's range is undefined behaviour.
 * Map anything beyond FLT_MAX to infinity, which is what pack 'f' does. */
static float
nv_to_float(NV v)
{
    if (v > FLT_MAX) {
        return (float)INFINITY;
    }
    if (v < -FLT_MAX) {
        return -(float)INFINITY;
    }
    return (float)v;
}

#define SvFLOAT_nomg(sv) nv_to_float(SvNV_nomg(sv))

/* Converts one innermost row of n elements into dst. */
static void
pack_row(pTHX_ AV *row, size_t n, elem_type type, char *dst, size_t *path, int depth, int *magic)
{
    size_t i;

    for (i = 0; i < n; i++) {
        SV *sv = NULL;
        int defined;

        /* Fast path for the common case of plain scalars in a plain array. */
        if (!*magic && !SvRMAGICAL(row)) {
            sv = AvARRAY(row)[i];
        }
        if (*magic || SvRMAGICAL(row) || (sv && SvGMAGICAL(sv))) {
            sv = fetch_elem(aTHX_ row, i, n, path, depth, magic);
        }
        defined = sv && SvOK(sv);

        if (defined && SvROK(sv)) {
            path[depth] = i;
            croak("Element %s is a reference, not a number", format_path(aTHX_ path, depth + 1));
        }
        /* Numifying a non-numeric string warns, and a __WARN__ handler is
         * Perl code too: keep the element alive and re-check rows from here
         * on. Numeric strings convert silently and need nothing. */
        if (defined && !SvNIOK(sv) && !looks_like_number(sv)) {
            sv_2mortal(SvREFCNT_inc_simple_NN(sv));
            *magic = 1;
        }

#define PACK_AS(ctype, conv) \
        { ctype v = defined ? (ctype)conv(sv) : (ctype)0; Copy(&v, dst + i * sizeof(ctype), 1, ctype); } break
        switch (type) {
        case ET_I8:  PACK_AS(I8,  SvIV_nomg);
        case ET_U8:  PACK_AS(U8,  SvUV_nomg);
        case ET_I16: PACK_AS(I16, SvIV_nomg);
        case ET_U16: PACK_AS(U16, SvUV_nomg);
        case ET_I32: PACK_AS(I32, SvIV_nomg);
        case ET_U32: PACK_AS(U32, SvUV_nomg);
        case ET_I64: PACK_AS(int64_t,  SvIV_nomg);
        case ET_U64: PACK_AS(uint64_t, SvUV_nomg);
        case ET_F32: PACK_AS(float,  SvFLOAT_nomg);
        case ET_F64: PACK_AS(double, SvNV_nomg);
        }
#undef PACK_AS
    }
}

static void
pack_level(pTHX_ AV *av, int level, int rank, const size_t *dims, elem_type type, size_t esize,
           char **dst, size_t *path, int *magic)
{
    size_t n = dims[level];
    size_t i;

    if ((size_t)av_count(av) != n) {
        croak("Row %s has %" UVuf " elements, expected %" UVuf,
              format_path(aTHX_ path, level), (UV)av_count(av), (UV)n);
    }
    if (level == rank - 1) {
        pack_row(aTHX_ av, n, type, *dst, path, level, magic);
        *dst += n * esize;
        return;
    }
    for (i = 0; i < n; i++) {
        path[level] = i;
        pack_level(aTHX_ elem_as_av(aTHX_ fetch_elem(aTHX_ av, i, n, path, level, magic), path, level + 1),
                   level + 1, rank, dims, type, esize, dst, path, magic);
    }
}

/* Shape from the first element at each level; pack_level checks the rest. */
static void
discover_dims(pTHX_ AV *av, int rank, size_t *dims, int *magic)
{
    size_t path[MAX_RANK] = { 0 };
    int level;

    for (level = 0; level < rank; level++) {
        dims[level] = (size_t)av_count(av);
        if (level == rank - 1) {
            break;
        }
        if (dims[level] == 0) {
            for (level++; level < rank; level++) {
                dims[level] = 0;
            }
            break;
        }
        av = elem_as_av(aTHX_ fetch_elem(aTHX_ av, 0, dims[level], path, level, magic), path, level + 1);
    }
}

static size_t
count_elems(pTHX_ const size_t *dims, int rank, size_t esize)
{
    size_t total = esize;
    int i;

    for (i = 0; i < rank; i++) {
        if (dims[i] != 0 && total > (size_t)-1 / dims[i]) {
            croak("Array is too large");
        }
        total *= dims[i];
    }
    return total;
}

static SV *
unpack_elem(pTHX_ elem_type type, const char *src)
{
#define UNPACK_AS(ctype, make) { ctype v; Copy(src, &v, 1, ctype); return make(v); }
    switch (type) {
    case ET_I8:  UNPACK_AS(I8,  newSViv);
    case ET_U8:  UNPACK_AS(U8,  newSVuv);
    case ET_I16: UNPACK_AS(I16, newSViv);
    case ET_U16: UNPACK_AS(U16, newSVuv);
    case ET_I32: UNPACK_AS(I32, newSViv);
    case ET_U32: UNPACK_AS(U32, newSVuv);
    case ET_I64: UNPACK_AS(int64_t,  newSViv);
    case ET_U64: UNPACK_AS(uint64_t, newSVuv);
    case ET_F32: UNPACK_AS(float,  newSVnv);
    case ET_F64: UNPACK_AS(double, newSVnv);
    }
#undef UNPACK_AS
    return newSV(0); /* not reached */
}

static void
set_elem(pTHX_ SV *sv, elem_type type, const char *src)
{
#define SET_AS(ctype, set) { ctype v; Copy(src, &v, 1, ctype); set(sv, v); } break
    switch (type) {
    case ET_I8:  SET_AS(I8,  sv_setiv);
    case ET_U8:  SET_AS(U8,  sv_setuv);
    case ET_I16: SET_AS(I16, sv_setiv);
    case ET_U16: SET_AS(U16, sv_setuv);
    case ET_I32: SET_AS(I32, sv_setiv);
    case ET_U32: SET_AS(U32, sv_setuv);
    case ET_I64: SET_AS(int64_t,  sv_setiv);
    case ET_U64: SET_AS(uint64_t, sv_setuv);
    case ET_F32: SET_AS(float,  sv_setnv);
    case ET_F64: SET_AS(double, sv_setnv);
    }
#undef SET_AS
    SvSETMAGIC(sv);
}

/* Builds a fresh nested array; nothing in here croaks, so nothing leaks. */
static AV *
build_level(pTHX_ int level, int rank, const size_t *dims, elem_type type, size_t esize, const char **src)
{
    AV *av = newAV();
    size_t n = dims[level];
    size_t i;
    SV **slots;

    if (n == 0) {
        return av;
    }
    av_extend(av, (SSize_t)n - 1);
    slots = AvARRAY(av);
    for (i = 0; i < n; i++) {
        if (level == rank - 1) {
            slots[i] = unpack_elem(aTHX_ type, *src);
            *src += esize;
        }
        else {
            slots[i] = newRV_noinc((SV *)build_level(aTHX_ level + 1, rank, dims, type, esize, src));
        }
    }
    AvFILLp(av) = (SSize_t)n - 1;
    return av;
}

/* Fills an existing nested array in place, reusing its scalars and rows and
 * resizing every level to the requested shape. */
static void
fill_level(pTHX_ AV *av, int level, int rank, const size_t *dims, elem_type type, size_t esize,
           const char **src, size_t *path)
{
    size_t n = dims[level];
    size_t i;

    av_fill(av, (SSize_t)n - 1);
    for (i = 0; i < n; i++) {
        SV *sv = *av_fetch(av, (SSize_t)i, TRUE);

        if (level == rank - 1) {
            set_elem(aTHX_ sv, type, *src);
            *src += esize;
            continue;
        }
        path[level] = i;
        SvGETMAGIC(sv);
        if (!SvOK(sv)) {
            sv_setrv_noinc(sv, (SV *)newAV());
            SvSETMAGIC(sv);
        }
        fill_level(aTHX_ elem_as_av(aTHX_ sv, path, level + 1), level + 1, rank, dims, type, esize, src, path);
    }
}

MODULE = CUDA::DriverAPI    PACKAGE = CUDA::DriverAPI

PROTOTYPES: DISABLE

SV *
_init(int ordinal)
CODE:
{
    CUdevice dev;
    CUcontext ctx;
    CUresult err;

    load_driver(aTHX);
    CHECK_CU(CU(cuInit)(0), "Init");
    CHECK_CU(CU(cuDeviceGet)(&dev, ordinal), "DeviceGet");
    CHECK_CU(CU(cuDevicePrimaryCtxRetain)(&ctx, dev), "DevicePrimaryCtxRetain");
    /* The caller never sees ctx if this fails, so give the reference back. */
    err = CU(cuCtxSetCurrent)(ctx);
    if (err != CUDA_SUCCESS) {
        CU(cuDevicePrimaryCtxRelease)(dev);
    }
    CHECK_CU(err, "CtxSetCurrent");

    RETVAL = newSViv(PTR2IV(ctx));
}
OUTPUT:
    RETVAL

void
_destroy(int ordinal)
CODE:
{
    CUdevice dev;

    load_driver(aTHX);
    CHECK_CU(CU(cuDeviceGet)(&dev, ordinal), "DeviceGet");
    CHECK_CU(CU(cuDevicePrimaryCtxRelease)(dev), "DevicePrimaryCtxRelease");
}

void
_synchronize(SV *ctx)
CODE:
{
    SET_CONTEXT(ctx);
    CHECK_CU(CU(cuCtxSynchronize)(), "CtxSynchronize");
}

UV
_byte_length(SV *data)
CODE:
{
    STRLEN len;

    (void)SvPVbyte(data, len);
    RETVAL = (UV)len;
}
OUTPUT:
    RETVAL

SV *
_malloc(SV *ctx, UV size)
CODE:
{
    CUdeviceptr dev_ptr;

    SET_CONTEXT(ctx);
    CHECK_CU(CU(cuMemAlloc)(&dev_ptr, (size_t)size), "MemAlloc");

    RETVAL = newSVuv((UV)dev_ptr);
}
OUTPUT:
    RETVAL

void
_memset(SV *ctx, SV *dev_ptr, unsigned int byte, UV size)
CODE:
{
    /* Ordered on the legacy default stream like kernel launches. */
    SET_CONTEXT(ctx);
    CHECK_CU(CU(cuMemsetD8Async)((CUdeviceptr)SvUV(dev_ptr), (unsigned char)byte, (size_t)size, NULL), "MemsetD8Async");
}

SV *
_write_array(SV *ctx, SV *dst_ptr, UV limit, SV *array, const char *type_name, int rank)
CODE:
{
    size_t esize;
    elem_type type = lookup_elem_type(aTHX_ type_name, &esize);
    size_t dims[MAX_RANK];
    size_t path[MAX_RANK] = { 0 };
    size_t bytes;
    char *staging;
    char *cursor;
    AV *dims_av;
    int i;
    int magic = 0;

    if (rank < 1 || rank > MAX_RANK) {
        croak("Invalid rank: %d (expected 1 to %d)", rank, MAX_RANK);
    }
    SvGETMAGIC(array);
    if (!SvROK(array) || SvTYPE(SvRV(array)) != SVt_PVAV) {
        croak("Expected an array reference");
    }
    /* Keep the array alive while element magic runs Perl code. */
    SvREFCNT_inc_simple_void_NN(SvRV(array));
    SAVEFREESV(SvRV(array));

    discover_dims(aTHX_ (AV *)SvRV(array), rank, dims, &magic);
    bytes = count_elems(aTHX_ dims, rank, esize);
    if (bytes > limit) {
        croak("Array of %" UVuf " bytes exceeds the buffer of %" UVuf " bytes", (UV)bytes, limit);
    }

    /* Freed when the caller's scope ends, also when a ragged row croaks. */
    Newx(staging, bytes ? bytes : 1, char);
    SAVEFREEPV(staging);
    cursor = staging;
    pack_level(aTHX_ (AV *)SvRV(array), 0, rank, dims, type, esize, &cursor, path, &magic);

    if (bytes) {
        SET_CONTEXT(ctx);
        CHECK_CU(CU(cuMemcpyHtoD)((CUdeviceptr)SvUV(dst_ptr), staging, bytes), "MemcpyHtoD");
    }

    dims_av = newAV();
    av_extend(dims_av, rank - 1);
    for (i = 0; i < rank; i++) {
        av_store(dims_av, i, newSVuv((UV)dims[i]));
    }
    RETVAL = newRV_noinc((SV *)dims_av);
}
OUTPUT:
    RETVAL

SV *
_read_array(SV *ctx, SV *src_ptr, UV limit, const char *type_name, AV *dims_av, SV *into)
CODE:
{
    size_t esize;
    elem_type type = lookup_elem_type(aTHX_ type_name, &esize);
    size_t dims[MAX_RANK];
    size_t path[MAX_RANK] = { 0 };
    int rank = (int)av_count(dims_av);
    size_t bytes;
    char *staging;
    const char *cursor;
    int i;

    if (rank < 1 || rank > MAX_RANK) {
        croak("dims must have 1 to %d sizes", MAX_RANK);
    }
    for (i = 0; i < rank; i++) {
        SV **v = av_fetch(dims_av, i, FALSE);
        dims[i] = v ? (size_t)SvUV(*v) : 0;
    }
    bytes = count_elems(aTHX_ dims, rank, esize);
    if (bytes > limit) {
        croak("Array of %" UVuf " bytes exceeds the buffer of %" UVuf " bytes", (UV)bytes, limit);
    }
    SvGETMAGIC(into);
    if (SvOK(into) && (!SvROK(into) || SvTYPE(SvRV(into)) != SVt_PVAV)) {
        croak("into must be an array reference");
    }

    Newx(staging, bytes ? bytes : 1, char);
    SAVEFREEPV(staging);
    if (bytes) {
        SET_CONTEXT(ctx);
        CHECK_CU(CU(cuMemcpyDtoH)(staging, (CUdeviceptr)SvUV(src_ptr), bytes), "MemcpyDtoH");
    }

    cursor = staging;
    if (SvOK(into)) {
        AV *target = (AV *)SvRV(into);

        /* Store magic on the elements may drop the caller's reference. */
        SvREFCNT_inc_simple_void_NN(target);
        SAVEFREESV(target);
        fill_level(aTHX_ target, 0, rank, dims, type, esize, &cursor, path);
        RETVAL = newRV_inc((SV *)target);
    }
    else {
        RETVAL = newRV_noinc((SV *)build_level(aTHX_ 0, rank, dims, type, esize, &cursor));
    }
}
OUTPUT:
    RETVAL

void
_free(SV *ctx, SV *dev_ptr)
CODE:
{
    SET_CONTEXT(ctx);
    CHECK_CU(CU(cuMemFree)((CUdeviceptr)SvUV(dev_ptr)), "MemFree");
}

void
_transfer_h2d(SV *ctx, SV *src, SV *dst_ptr, UV limit)
CODE:
{
    STRLEN len;
    const char *buf = SvPVbyte(src, len);

    check_bounds(aTHX_ len, limit);
    SET_CONTEXT(ctx);
    CHECK_CU(CU(cuMemcpyHtoD)((CUdeviceptr)SvUV(dst_ptr), buf, len), "MemcpyHtoD");
}

void
_transfer_d2h(SV *ctx, SV *src_ptr, SV *dst, UV limit)
CODE:
{
    STRLEN len;
    char *buf;

    if (!sv_utf8_downgrade(dst, TRUE)) {
        croak("Destination buffer must be a byte string");
    }
    buf = SvPV_force(dst, len);
    check_bounds(aTHX_ len, limit);

    SET_CONTEXT(ctx);
    CHECK_CU(CU(cuMemcpyDtoH)(buf, (CUdeviceptr)SvUV(src_ptr), len), "MemcpyDtoH");
    SvSETMAGIC(dst);
}

SV *
_read(SV *ctx, SV *src_ptr, UV size)
CODE:
{
    /* Mortal until the copy succeeds, so that a croak does not leak it. */
    SV *buf = new_buffer(aTHX_ size);

    SET_CONTEXT(ctx);
    CHECK_CU(CU(cuMemcpyDtoH)(SvPVX(buf), (CUdeviceptr)SvUV(src_ptr), (size_t)size), "MemcpyDtoH");
    set_buffer_length(aTHX_ buf, size);

    RETVAL = SvREFCNT_inc_simple_NN(buf);
}
OUTPUT:
    RETVAL

SV *
_load_library(SV *ctx, const char *path)
CODE:
{
    CUlibrary library;

    SET_CONTEXT(ctx);
    CHECK_CU(CU(cuLibraryLoadFromFile)(&library, path, NULL, NULL, 0, NULL, NULL, 0), "LibraryLoadFromFile");

    RETVAL = newSViv(PTR2IV(library));
}
OUTPUT:
    RETVAL

SV *
_load_library_data(SV *ctx, SV *image)
CODE:
{
    CUlibrary library;
    /* PTX must be NUL-terminated; byte-string SVs always are. */
    const char *code = SvPVbyte_nolen(image);

    SET_CONTEXT(ctx);
    CHECK_CU(CU(cuLibraryLoadData)(&library, code, NULL, NULL, 0, NULL, NULL, 0), "LibraryLoadData");

    RETVAL = newSViv(PTR2IV(library));
}
OUTPUT:
    RETVAL

void
_compute_capability(SV *ctx, int ordinal)
PPCODE:
{
    CUdevice dev;
    int major, minor;

    SET_CONTEXT(ctx);
    CHECK_CU(CU(cuDeviceGet)(&dev, ordinal), "DeviceGet");
    CHECK_CU(CU(cuDeviceGetAttribute)(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "DeviceGetAttribute");
    CHECK_CU(CU(cuDeviceGetAttribute)(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "DeviceGetAttribute");

    EXTEND(SP, 2);
    mPUSHi(major);
    mPUSHi(minor);
}

void
_nvrtc_supported_archs()
PPCODE:
{
    int n, i;
    int *archs;

    CHECK_NVRTC(nvrtcGetNumSupportedArchs(&n), "GetNumSupportedArchs");
    Newx(archs, n ? n : 1, int);
    SAVEFREEPV(archs);
    CHECK_NVRTC(nvrtcGetSupportedArchs(archs), "GetSupportedArchs");

    EXTEND(SP, n);
    for (i = 0; i < n; i++) {
        mPUSHi(archs[i]);
    }
}

void
_compile(const char *source, const char *name, AV *options)
PPCODE:
{
    nvrtcProgram prog;
    nvrtcResult res;
    SSize_t nopts = av_count(options);
    SSize_t i;
    const char **opts;
    size_t log_size = 0;
    size_t image_size = 0;
    SV *log;
    SV *image = NULL;

    /* Everything that can croak happens before or after the program exists. */
    Newxz(opts, nopts ? nopts : 1, const char *);
    SAVEFREEPV(opts);
    for (i = 0; i < nopts; i++) {
        SV **v = av_fetch(options, i, FALSE);
        opts[i] = v ? SvPV_nolen(*v) : "";
    }
    log = new_buffer(aTHX_ 0);

    CHECK_NVRTC(nvrtcCreateProgram(&prog, source, name, 0, NULL, NULL), "CreateProgram");
    res = nvrtcCompileProgram(prog, (int)nopts, opts);

    if (nvrtcGetProgramLogSize(prog, &log_size) == NVRTC_SUCCESS && log_size > 1) {
        SvGROW(log, log_size);
        if (nvrtcGetProgramLog(prog, SvPVX(log)) == NVRTC_SUCCESS) {
            set_buffer_length(aTHX_ log, log_size - 1);
        }
    }

    if (res == NVRTC_SUCCESS) {
        /* A real architecture (sm_XY) yields a cubin; a virtual one (compute_XY)
         * only PTX, whose size includes the trailing NUL. */
        res = nvrtcGetCUBINSize(prog, &image_size);
        if (res == NVRTC_SUCCESS && image_size > 0) {
            image = new_buffer(aTHX_ image_size);
            res = nvrtcGetCUBIN(prog, SvPVX(image));
        }
        else if (res == NVRTC_SUCCESS && (res = nvrtcGetPTXSize(prog, &image_size)) == NVRTC_SUCCESS) {
            image = new_buffer(aTHX_ image_size);
            res = nvrtcGetPTX(prog, SvPVX(image));
            image_size--;
        }
    }
    nvrtcDestroyProgram(&prog);

    if (res != NVRTC_SUCCESS) {
        croak("NVRTC error at CompileProgram: %s (%d)\n%" SVf,
              nvrtcGetErrorString(res), (int)res, SVfARG(log));
    }
    set_buffer_length(aTHX_ image, image_size);

    EXTEND(SP, 2);
    PUSHs(image);
    PUSHs(log);
}

void
_unload_library(SV *ctx, SV *library)
CODE:
{
    SET_CONTEXT(ctx);
    CHECK_CU(CU(cuLibraryUnload)(INT2PTR(CUlibrary, SvIV(library))), "LibraryUnload");
}

SV *
_get_function(SV *ctx, SV *library, const char *name)
CODE:
{
    CUkernel kernel;
    CUfunction func;

    SET_CONTEXT(ctx);
    CHECK_CU(CU(cuLibraryGetKernel)(&kernel, INT2PTR(CUlibrary, SvIV(library)), name), "LibraryGetKernel");
    CHECK_CU(CU(cuKernelGetFunction)(&func, kernel), "KernelGetFunction");

    RETVAL = newSViv(PTR2IV(func));
}
OUTPUT:
    RETVAL

SV *
_param_sizes(SV *ctx, SV *function)
CODE:
{
    AV *sizes;
    size_t i;

    SET_CONTEXT(ctx);
    if (!p_func_get_param_info) {
        XSRETURN_UNDEF;
    }
    sizes = (AV *)sv_2mortal((SV *)newAV());
    for (i = 0;; i++) {
        size_t offset, size;
        CUresult err = p_func_get_param_info(INT2PTR(CUfunction, SvIV(function)), i, &offset, &size);

        if (err == CUDA_ERROR_INVALID_VALUE) {
            break; /* past the last parameter */
        }
        if (err != CUDA_SUCCESS) {
            XSRETURN_UNDEF; /* the check is optional: launch unchecked */
        }
        av_push(sizes, newSVuv((UV)size));
    }
    /* INVALID_VALUE at index 0 may also be a failure; extra arguments to a
     * kernel without parameters are harmless anyway. */
    if (av_count(sizes) == 0) {
        XSRETURN_UNDEF;
    }
    RETVAL = newRV_inc((SV *)sizes);
}
OUTPUT:
    RETVAL

void
_launch(SV *ctx, SV *function, AV *args, AV *config)
CODE:
{
    void **params;
    unsigned int cnf[LAUNCH_CONFIG_SIZE] = { 1, 1, 1, 1, 1, 1, 0 };
    SSize_t nargs = av_count(args);
    SSize_t ncnf = av_count(config);
    SSize_t i;

    if (ncnf > LAUNCH_CONFIG_SIZE) {
        croak("Too many launch config values (%d > %d)", (int)ncnf, LAUNCH_CONFIG_SIZE);
    }
    for (i = 0; i < ncnf; i++) {
        SV **v = av_fetch(config, i, FALSE);
        if (v) {
            cnf[i] = (unsigned int)SvUV(*v);
        }
    }

    /* Each element of args is a kernel argument already packed into its
     * native byte representation; cuLaunchKernel copies them at launch time.
     * The array is freed when the caller's scope ends. */
    Newxz(params, nargs ? nargs : 1, void *);
    SAVEFREEPV(params);
    for (i = 0; i < nargs; i++) {
        SV **v = av_fetch(args, i, FALSE);
        if (!v) {
            croak("Kernel argument %d is undefined", (int)i);
        }
        params[i] = SvPVbyte_nolen(*v);
    }

    /* Launched on the legacy default stream without waiting: later copies,
     * frees and synchronize() order themselves after the kernel. */
    SET_CONTEXT(ctx);
    CHECK_CU(CU(cuLaunchKernel)(INT2PTR(CUfunction, SvIV(function)),
                            cnf[0], cnf[1], cnf[2], cnf[3], cnf[4], cnf[5], cnf[6],
                            NULL, params, NULL), "LaunchKernel");
}

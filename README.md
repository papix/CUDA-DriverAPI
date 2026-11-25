# NAME

CUDA::DriverAPI - CUDA bindings for Perl using CUDA Driver API

# SYNOPSIS

    use CUDA::DriverAPI;

    my $ctx = CUDA::DriverAPI->new();

    my $program = $ctx->compile(<<'CUDA');
    extern "C" __global__ void axpy(float a, const float *x, float *y, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a * x[i] + y[i];
    }
    CUDA

    my $n = 1000;
    my $x = $ctx->buffer(data => pack('f*', 1 .. $n));
    my $y = $ctx->buffer(size => 4 * $n);
    $y->write(pack('f*', (1) x $n));

    $ctx->run($program, 'axpy', [ [ f32 => 2 ], $x, $y, [ i32 => $n ] ],
              [ int(($n + 255) / 256), 1, 1, 256 ]);

    my @result = unpack('f*', $y->read);    # 2 * x + 1

    # Buffers are freed when they go out of scope.

# DESCRIPTION

CUDA::DriverAPI is CUDA bindings module for Perl.

It uses the device's primary context (shared with the CUDA Runtime API),
loads kernels with the context-independent `cuLibrary*` API, and compiles
CUDA C++ source at run time with NVRTC.

# METHODS

- `CUDA::DriverAPI->new(%args)`

    Retains the primary context of a device. `device` is the device ordinal
    (default: 0).

- `$ctx->buffer(data => $bytes)`
- `$ctx->buffer(size => $size)`

    Allocates device memory and returns a [CUDA::DriverAPI::Buffer](https://metacpan.org/pod/CUDA%3A%3ADriverAPI%3A%3ABuffer), which frees
    the memory when it goes out of scope. With `data`, the memory is
    `length($bytes)` bytes and initialized with `$bytes`; with `size`, it is
    `$size` bytes and uninitialized.

- `$ctx->compile($source, %args)`

    Compiles CUDA C++ source with NVRTC and returns a [CUDA::DriverAPI::Program](https://metacpan.org/pod/CUDA%3A%3ADriverAPI%3A%3AProgram)
    to pass to `run`. Kernels must be declared `extern "C"` to be found by name.

    `name` is the file name shown in compiler messages (default: `kernel.cu`).
    `options` is an array reference of NVRTC options such as `-std=c++20`. Unless
    it contains `--gpu-architecture`, the code is compiled for the device's own
    architecture. The headers of the CUDA Toolkit the module was built with
    (e.g. `cuda_fp16.h`, `cooperative_groups.h`, `cuda/std/...`) are found
    without options; directories given with `-I` are searched first. (For a
    toolkit installed into `/usr` by distribution packages only `/usr/include/cccl`
    is added, since `/usr/include` also holds host headers; pass `-I/usr/include`
    yourself if a kernel needs headers from there.) A compile error croaks with the
    compiler log.

- `$ctx->load($image)`

    Loads a compiled image (cubin, fatbin or PTX, e.g. a saved
    `$program->image`) and returns a [CUDA::DriverAPI::Program](https://metacpan.org/pod/CUDA%3A%3ADriverAPI%3A%3AProgram), like
    `compile` but without NVRTC. Use it to cache compiled code across processes,
    keyed by `compute_capability`. The driver loads code lazily, so an invalid
    image is reported by the first `run` that uses it.

- `$ctx->compute_capability`

    Returns the device's compute capability as `($major, $minor)`, e.g.
    `(8, 9)`.

- `$ctx->run($code, $function, \@args, \@config)`

    Launches the kernel `$function` in `$code`, which is either a program from
    `compile` or the path of a PTX, cubin or fatbin file. Files and functions are
    looked up once and cached.

    Each element of `@args` is a [CUDA::DriverAPI::Buffer](https://metacpan.org/pod/CUDA%3A%3ADriverAPI%3A%3ABuffer), a device pointer, or
    a `[ $type => $value ]` pair passed by value, where `$type` is one of
    `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32`, `f64`
    and `ptr`. A buffer must come from this object and not be freed yet.

    `@config` is
    `(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes)`;
    omitted values default to `(1, 1, 1, 1, 1, 1, 0)`.

    The launch is asynchronous: `run` returns without waiting for the kernel.
    Reading memory back, freeing it, `destroy` and `synchronize` wait for it, and
    an error raised by the kernel is reported by the next of these calls.

- `$ctx->synchronize`

    Waits until all launched kernels have finished.

- `$ctx->destroy`

    Waits for pending work, frees all memory and loaded code, and releases the
    primary context. Called automatically when the object and every buffer and
    program created from it have gone out of scope.

## Raw device pointers

These methods work with plain device pointers instead of buffer objects.
Memory allocated by `malloc` is not freed automatically before `destroy`.

- `$ctx->malloc($bytes)`

    Allocates `length($bytes)` bytes of device memory and returns the device
    pointer. The content is not copied; use `transfer_h2d`.

- `$ctx->free($ptr)`

    Frees device memory allocated by `malloc`.

- `$ctx->transfer_h2d($bytes, $ptr)`

    Copies `$bytes` to the device memory `$ptr`.

- `$ctx->transfer_d2h($ptr [, $size])`

    Returns `$size` bytes copied from the device memory `$ptr`. `$size` defaults
    to the size of the allocation.

- `$ctx->transfer_d2h($ptr, \$buffer)`

    Copies `length($buffer)` bytes from the device memory `$ptr` into `$buffer`.

Transfers into and out of memory allocated by `malloc` croak when they would
run past the end of the allocation.

# PERFORMANCE

- `run`, `fill` and the copies queued before them do not wait for the GPU.
Launch as many kernels as needed and read the results once at the end;
reading waits for everything launched before it.
- Use `write_array` and `read_array` for nested Perl arrays instead of
`pack`/`unpack`: they convert in C and avoid an intermediate string. Pass
`into` to `read_array` in loops to reuse the same Perl array.
- Use `fill` instead of writing a string of zeros to initialize output buffers.
- NVRTC takes a noticeable time on the first compile. Save `$program->image`
to a file keyed by `compute_capability` and `load` it in later runs; see
`eg/matmul.pl`.

# ERRORS

Every method croaks on failure. Errors from the driver read
`CUDA error at <call>: <name> (<code>): <description>`,
e.g. `CUDA error at LibraryGetKernel: CUDA_ERROR_NOT_FOUND (500): named
symbol not found`, and NVRTC compile errors include the compiler log.

Because launches are asynchronous, an error raised while a kernel runs (such
as an illegal memory access) is reported by the next call that waits for the
GPU. Such errors are sticky: the device's primary context, shared by every
object for that device, stays unusable for the rest of the process.

# THREADS

Objects are not cloned into new Perl ithreads; create a new object in each
thread instead.

# INSTALLATION

Requirements: Perl 5.40 or later, an NVIDIA driver, and the CUDA Toolkit 12.0
or later including NVRTC (`cuda.h`, `nvrtc.h` and `libnvrtc`).

    cpanm CUDA::DriverAPI

or from a checkout:

    cpanm --with-configure --installdeps .
    perl Build.PL
    ./Build
    ./Build test
    ./Build install

The toolkit is found without configuration in the usual places, in this
order: the directory of `nvcc` on `PATH`, `/usr/local/cuda`, the newest
`/usr/local/cuda-X.Y`, `/opt/cuda`, `$CONDA_PREFIX`, and `/usr`
(distribution packages). `perl Build.PL` prints the one it uses. To choose
another, set `CUDA_HOME` (or `CUDA_PATH`) to the directory that contains
`include/cuda.h`; it is then the only place looked at.

On WSL, `libcuda` is taken from `/usr/lib/wsl/lib`;
install only the toolkit (e.g. the `cuda-toolkit-13-3` package from NVIDIA's
`wsl-ubuntu` repository), not a Linux driver. Tests that need a GPU are
skipped when no CUDA device is available.

# EXAMPLES

- `eg/axpy.pl`

    Compiles a kernel, moves data with `write_array`/`read_array` and launches it.

- `eg/matmul.pl`

    Matrix multiplication with the compiled kernel cached on disk.

- `author/bench-array.pl`

    Compares `write_array`/`read_array` with `pack`/`unpack`.

# SEE ALSO

[CUDA::DriverAPI::Buffer](https://metacpan.org/pod/CUDA%3A%3ADriverAPI%3A%3ABuffer), [CUDA::DriverAPI::Program](https://metacpan.org/pod/CUDA%3A%3ADriverAPI%3A%3AProgram),
[CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/),
[NVRTC](https://docs.nvidia.com/cuda/nvrtc/)

# LICENSE

Copyright (C) papix.

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

# AUTHOR

papix <mail@papix.net>

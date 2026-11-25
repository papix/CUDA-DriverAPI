use v5.40;
use Test2::V0;
use File::Temp qw(tempfile);

use CUDA::DriverAPI;

skip_all "CUDA device is not available: $@" unless eval { CUDA::DriverAPI->new };

my $ctx = CUDA::DriverAPI->new;

my $source = <<'CUDA';
extern "C" __global__ void scale(float factor, float *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= factor;
    }
}
CUDA

subtest 'compile and run' => sub {
    my $program = $ctx->compile($source);
    isa_ok $program, 'CUDA::DriverAPI::Program';
    is $program->log, '';

    my $n    = 100;
    my $data = $ctx->buffer(data => pack('f*', 1 .. $n));
    $ctx->run($program, 'scale', [ [ f32 => 0.5 ], $data, [ i32 => $n ] ], [ 1, 1, 1, 128 ]);

    is [ unpack('f*', $data->read) ], [ map { $_ / 2 } 1 .. $n ];
};

subtest 'compiled for the device architecture by default' => sub {
    my ($major, $minor) = $ctx->compute_capability;
    my $program = $ctx->compile($source);

    # A cubin is an ELF file; PTX is text.
    is substr($program->image, 0, 4), "\x7fELF", 'cubin';
    like $ctx->_default_arch, qr/\A--gpu-architecture=(?:sm|compute)_\d+\z/;
    note "compute capability $major.$minor: " . $ctx->_default_arch;
};

subtest 'options' => sub {
    my $program = $ctx->compile(
        "#ifndef FACTOR\n#error FACTOR\n#endif\n$source",
        name    => 'scale.cu',
        options => [ '-DFACTOR=1', '-std=c++20', '--gpu-architecture=compute_75' ],
    );
    like $program->image, qr/\.version \d+\.\d+/, 'a virtual architecture yields PTX';

    my $data = $ctx->buffer(data => pack('f', 3));
    $ctx->run($program, 'scale', [ [ f32 => 2 ], $data, [ i32 => 1 ] ]);
    is unpack('f', $data->read), 6;
};

subtest 'the image can be saved and run from a file' => sub {
    my ($fh, $file) = tempfile(SUFFIX => '.cubin', UNLINK => 1);
    binmode $fh;
    print {$fh} $ctx->compile($source)->image;
    close $fh;

    my $data = $ctx->buffer(data => pack('f', 5));
    $ctx->run($file, 'scale', [ [ f32 => 3 ], $data, [ i32 => 1 ] ]);
    is unpack('f', $data->read), 15;
};

subtest 'compile errors carry the log' => sub {
    my $error = dies { $ctx->compile("__global__ void broken() { undefined_symbol; }", name => 'broken.cu') };
    like $error, qr/NVRTC error at CompileProgram: NVRTC_ERROR_COMPILATION/;
    like $error, qr/broken\.cu.*undefined_symbol/s;
};

subtest 'toolkit headers are found without options' => sub {
    my $program = $ctx->compile(<<'CUDA');
#include <cuda_fp16.h>
#include <cuda/std/cstdint>
extern "C" __global__ void halve(float *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuda::std::int32_t unused = 0;
        data[i] = __half2float(__float2half(data[i]) * __float2half(0.5f)) + unused;
    }
}
CUDA
    my $data = $ctx->buffer(data => pack('f*', 2, 4, 6));
    $ctx->run($program, 'halve', [ $data, [ i32 => 3 ] ], [ 1, 1, 1, 32 ]);
    is [ unpack('f*', $data->read) ], [ 1, 2, 3 ];
};

subtest 'warnings are kept in the log' => sub {
    my $program = $ctx->compile(qq{#warning "kept in the log"\nextern "C" __global__ void k() {}\n});
    like $program->log, qr/kept in the log/;
};

subtest 'unloaded when it goes out of scope' => sub {
    my $library;
    {
        my $program = $ctx->compile($source);
        $library = $program->{library};
        ok exists $ctx->{libraries}{"program:$library"};
    }
    ok !exists $ctx->{libraries}{"program:$library"};
};

subtest 'destroy unloads programs that are still alive' => sub {
    my $scoped  = CUDA::DriverAPI->new;
    my $program = $scoped->compile($source);

    ok $scoped->destroy;
    ok lives { undef $program }, 'releasing afterwards is a no-op';
};

subtest 'programs belong to their context' => sub {
    my $other   = CUDA::DriverAPI->new;
    my $program = $other->compile($source);
    like dies { $ctx->run($program,                'scale', []) }, qr/belongs to another CUDA::DriverAPI object/;
    like dies { $ctx->run($ctx->buffer(size => 1), 'scale', []) }, qr/Not a compiled program/;
};

done_testing;

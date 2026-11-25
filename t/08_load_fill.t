use v5.40;
use Test2::V0;

use CUDA::DriverAPI;

skip_all "CUDA device is not available: $@" unless eval { CUDA::DriverAPI->new };

my $ctx = CUDA::DriverAPI->new;

my $source = <<'CUDA';
extern "C" __global__ void add_one(int *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1;
}
CUDA

subtest 'fill' => sub {
    my $buffer = $ctx->buffer(data => 'x' x 16);

    is $buffer->fill,            $buffer,   'returns the buffer';
    is $buffer->read,            "\0" x 16, 'zero by default';
    is $buffer->fill(255)->read, "\xff" x 16;

    like dies { $buffer->fill(256) }, qr/Invalid byte: 256/;
    like dies { $buffer->fill(-1) },  qr/Invalid byte: -1/;
};

subtest 'fill is ordered with kernels' => sub {
    my $program = $ctx->compile($source);
    my $buffer  = $ctx->buffer(size => 4 * 64);

    $buffer->fill;
    $ctx->run($program, 'add_one', [ $buffer, [ i32 => 64 ] ], [ 1, 1, 1, 64 ]);
    $buffer->fill;
    $ctx->run($program, 'add_one', [ $buffer, [ i32 => 64 ] ], [ 1, 1, 1, 64 ]);
    is $buffer->read_array(type => 'i32', dims => [64]), [ (1) x 64 ];
};

subtest 'compute_capability' => sub {
    my ($major, $minor) = $ctx->compute_capability;
    like $major, qr/\A[1-9][0-9]*\z/;
    like $minor, qr/\A[0-9]+\z/;
};

subtest 'load a cached image in another context' => sub {
    my $image = $ctx->compile($source)->image;

    my $other   = CUDA::DriverAPI->new;
    my $program = $other->load($image);
    isa_ok $program, 'CUDA::DriverAPI::Program';
    is $program->log,   '';
    is $program->image, $image;

    my $buffer = $other->buffer(size => 16)->fill;
    $other->run($program, 'add_one', [ $buffer, [ i32 => 4 ] ], [ 1, 1, 1, 4 ]);
    is $buffer->read_array(type => 'i32', dims => [4]), [ 1, 1, 1, 1 ];
};

subtest 'load PTX' => sub {
    my $ptx     = $ctx->compile($source, options => ['--gpu-architecture=compute_75'])->image;
    my $program = $ctx->load($ptx);

    my $buffer = $ctx->buffer(size => 4)->fill;
    $ctx->run($program, 'add_one', [ $buffer, [ i32 => 1 ] ]);
    is $buffer->read_array(type => 'i32', dims => [1]), [1];
};

subtest 'invalid image' => sub {

    # The driver loads code lazily, so a broken image is reported when its
    # first kernel is looked up.
    like dies { $ctx->run($ctx->load('not an image'), 'add_one', []) },
        qr/CUDA error at Library\w+: CUDA_ERROR_INVALID_IMAGE/;
};

done_testing;

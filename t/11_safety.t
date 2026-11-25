use v5.40;
use Test2::V0;

use File::Temp ();
use CUDA::DriverAPI;

skip_all "CUDA device is not available: $@" unless eval { CUDA::DriverAPI->new };

my $ctx = CUDA::DriverAPI->new;

my $program = $ctx->compile(<<'CUDA');
extern "C" __global__ void scale(float *data, int n, double factor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= (float)factor;
}
CUDA

subtest 'matching kernel arguments run' => sub {
    my $buffer = $ctx->buffer(size => 8);
    $buffer->write_array([ 1, 2 ], type => 'f32');

    $ctx->run($program, 'scale', [ $buffer, [ i32 => 2 ], [ f64 => 3 ] ], [ 1, 1, 1, 32 ]);
    is $buffer->read_array(type => 'f32', dims => [2]), [ 3, 6 ];
};

subtest 'kernel arguments are checked against the kernel' => sub {

    # cuFuncGetParamInfo needs a CUDA 12.4 driver; older ones launch unchecked.
    my $function = CUDA::DriverAPI::_get_function($ctx->_context, $ctx->_resolve_library($program), 'scale');
    skip_all 'the driver cannot report kernel parameters'
        unless defined CUDA::DriverAPI::_param_sizes($ctx->_context, $function);

    my $buffer = $ctx->buffer(size => 8);
    $buffer->write_array([ 1, 2 ], type => 'f32');

    like dies { $ctx->run($program, 'scale', [ $buffer, [ i32 => 2 ] ]) }, qr/Kernel scale takes 3 arguments, got 2/;
    like dies { $ctx->run($program, 'scale', [ $buffer, [ i32 => 2 ], [ f32 => 3 ] ]) },
        qr/Argument 2 of scale is 4 bytes, the kernel parameter is 8 bytes/;
    is $buffer->read_array(type => 'f32', dims => [2]), [ 1, 2 ], 'nothing ran';

    # As before: the driver reads the first bytes of a longer argument and
    # ignores extra ones.
    ok lives { $ctx->run($program, 'scale', [ $buffer, [ i64 => 2 ], [ f64 => 3 ], [ i32 => 0 ] ], [ 1, 1, 1, 32 ]) };
    is $buffer->read_array(type => 'f32', dims => [2]), [ 3, 6 ], 'longer and extra arguments run';
};

subtest 'empty arrays have a bounded number of rows' => sub {
    my $buffer = $ctx->buffer(size => 4);

    is $buffer->read_array(type => 'f32', dims => [ 3, 0 ]),                    [ [], [], [] ], 'a few empty rows';
    is scalar @{ $buffer->read_array(type => 'f32', dims => [ 100, 100, 0 ]) }, 100, 'up to 65536 for a small buffer';
    is $buffer->read_array(type => 'f32', dims => [ 0, 1_000_000_000 ]),        [],  'no rows';
    like dies { $buffer->read_array(type => 'f32', dims => [ 1_000_000_000, 0 ]) },
        qr/dims \[1000000000 0\] describe 1000000000 empty rows, more than the limit of 65536/;
    like dies { $buffer->read_array(type => 'f32', dims => [ 1_000_000_000, 0, 5 ]) },
        qr/describe 1000000000 empty rows/, 'a zero in the middle';

    my $large = $ctx->buffer(size => 100_000);
    is scalar @{ $large->read_array(type => 'u8', dims => [ 100_000, 0 ]) }, 100_000, 'one row per byte';
};

subtest 'a forked child leaves the parent\'s objects alone' => sub {
    my $buffer = $ctx->buffer(size => 8);
    $buffer->write_array([ 1, 2 ], type => 'f32');
    my $stderr = File::Temp->new;

    my $pid = fork // die "fork: $!";
    if ($pid == 0) {
        open STDERR, '>&', $stderr or die $!;
        my $error = dies { $buffer->read };
        print STDERR "unexpected: ", $error // 'no error', "\n"
            unless ($error // '') =~ /CUDA objects cannot be used in a forked child process/;
        undef $buffer;
        undef $program;
        $ctx->destroy;
        exit 0;
    }
    waitpid $pid, 0;
    is $?,                                                                    0,  'the child exited cleanly';
    is do { local $/; open my $fh, '<', $stderr->filename or die $!; <$fh> }, '', 'nothing on its STDERR';

    is $buffer->read_array(type => 'f32', dims => [2]), [ 1, 2 ], 'the parent\'s buffer still works';
    $ctx->run($program, 'scale', [ $buffer, [ i32 => 2 ], [ f64 => 2 ] ], [ 1, 1, 1, 32 ]);
    is $buffer->read_array(type => 'f32', dims => [2]), [ 2, 4 ], 'and so does its program';
};

done_testing;

#!/usr/bin/env perl
# y = a * x + y on the GPU: compile a kernel, move data, launch, read back.
#
#   perl -Mblib eg/axpy.pl [n]
use v5.40;
use CUDA::DriverAPI;

my $n = shift // 1_000_000;

my $ctx     = CUDA::DriverAPI->new;
my $program = $ctx->compile(<<'CUDA');
extern "C" __global__ void axpy(float a, const float *x, float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
CUDA

my $x = $ctx->buffer(size => 4 * $n);
my $y = $ctx->buffer(size => 4 * $n);
$x->write_array([ 1 .. $n ],  type => 'f32');
$y->write_array([ (1) x $n ], type => 'f32');

my $threads = 256;
my @args    = ([ f32 => 2 ], $x, $y, [ i32 => $n ]);
my @config  = (int(($n + $threads - 1) / $threads), 1, 1, $threads);
$ctx->run($program, 'axpy', \@args, \@config);

my $result = $y->read_array(type => 'f32', dims => [$n]);
say "y[0] = $result->[0], y[-1] = $result->[-1]";    # 2 * x + 1

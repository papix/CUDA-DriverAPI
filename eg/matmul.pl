#!/usr/bin/env perl
# Multiplies two n x n matrices on the GPU. The kernel is compiled with NVRTC
# once per source and device architecture and cached on disk, so later runs
# skip NVRTC.
#
#   perl -Mblib eg/matmul.pl [n] [cache-dir]
#
# The cache directory defaults to $XDG_CACHE_HOME/cuda-driverapi (or
# ~/.cache/cuda-driverapi). Keep it private: whatever is cached there runs on
# the GPU.
use v5.40;
use Digest::SHA qw(sha256_hex);
use File::Path  qw(make_path);
use File::Spec;
use File::Temp;
use Time::HiRes qw(time);
use CUDA::DriverAPI;

my $n         = shift // 1024;
my $cache_dir = shift // File::Spec->catdir($ENV{XDG_CACHE_HOME} // "$ENV{HOME}/.cache", 'cuda-driverapi');

my $SOURCE = <<'CUDA';
extern "C" __global__ void matmul(const float *a, const float *b, float *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0;
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}
CUDA

my $ctx = CUDA::DriverAPI->new;

# Key the cache by the source and the compute capability: a cubin only runs
# on its architecture, and an edited kernel must not reuse the old binary.
my $cache = File::Spec->catfile($cache_dir,
    sprintf('matmul-%s-sm_%d%d.cubin', substr(sha256_hex($SOURCE), 0, 16), $ctx->compute_capability));

sub compile_and_cache () {
    my $program = $ctx->compile($SOURCE);

    # Write to a temporary file and rename it, so that a concurrent run never
    # reads a half-written cubin.
    make_path($cache_dir, { mode => 0700 });
    my $tmp = File::Temp->new(DIR => $cache_dir, UNLINK => 0);
    binmode $tmp;
    print {$tmp} $program->image;
    close $tmp or die "$tmp: $!";
    rename "$tmp", $cache or die "$cache: $!";
    return $program;
}

sub cached_program () {
    open my $fh, '<:raw', $cache or return;
    return $ctx->load(do { local $/; <$fh> });
}

my $lhs = [
    map {
        my $i = $_;
        [ map { ($i + $_) % 7 } 0 .. $n - 1 ]
    } 0 .. $n - 1
];
my $rhs = [
    map {
        my $i = $_;
        [ map { ($i * $_) % 5 } 0 .. $n - 1 ]
    } 0 .. $n - 1
];

my $start = time;
my $buf_a = $ctx->buffer(size => 4 * $n * $n);
my $buf_b = $ctx->buffer(size => 4 * $n * $n);
my $buf_c = $ctx->buffer(size => 4 * $n * $n);
$buf_a->write_array($lhs, type => 'f32', rank => 2);
$buf_b->write_array($rhs, type => 'f32', rank => 2);

my @args   = ($buf_a, $buf_b, $buf_c, [ i32 => $n ]);
my @config = (int(($n + 15) / 16), int(($n + 15) / 16), 1, 16, 16, 1);

# A broken cache may fail when it is loaded or, because the driver loads code
# lazily, only when the kernel is first used; recompile in either case.
my $from = -e $cache ? 'cache' : 'NVRTC';
my $program;
try {
    $program = $from eq 'cache' ? cached_program() : compile_and_cache();
    $ctx->run($program, 'matmul', \@args, \@config);
}
catch ($e) {
    die $e if $from eq 'NVRTC';
    warn "Discarding the cached kernel: $e";
    ($program, $from) = (compile_and_cache(), 'NVRTC');
    $ctx->run($program, 'matmul', \@args, \@config);
}

my $c = $buf_c->read_array(type => 'f32', dims => [ $n, $n ]);
printf "%dx%d matmul (kernel from %s): %.1f ms\n", $n, $n, $from, 1000 * (time - $start);

# Spot-check a few entries against Perl.
for my $i (0, int($n / 2), $n - 1) {
    my $expected = 0;
    $expected += $lhs->[$i][$_] * $rhs->[$_][$i] for 0 .. $n - 1;
    die "c[$i][$i] = $c->[$i][$i], expected $expected\n" if $c->[$i][$i] != $expected;
}
say 'ok';

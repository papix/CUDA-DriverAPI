use strict;
use Test::More;

use CUDA::DeviceAPI;
use CUDA::DeviceAPI::Array;
use File::Spec;
use Cwd;

my $max = 10;
my $path = File::Spec->catfile(cwd(), qw/ t ptx 01_simple.ptx /);
my $host_data = CUDA::DeviceAPI::Array->new([ 1..$max ]);

my $ctx = CUDA::DeviceAPI->new();
$ctx->init();

my $dev_ptr_1 = $ctx->malloc_from($host_data => 'f');
my $dev_ptr_2 = $ctx->malloc_from($host_data => 'f');
my $dev_ptr_3 = $ctx->malloc($host_data->size() => 'f');

$ctx->run($path, 'kernel_sum', [
    $dev_ptr_1 => 'p',
    $dev_ptr_2 => 'p',
    $dev_ptr_3 => 'p',
    $max       => 'i',
], [
    $max
]
);

$ctx->transfer_d2h($dev_ptr_3, \my $result);

my @results = unpack 'f*', $result;
for my $i (0..$max - 1) {
    is($results[$i], ($i + 1)*2);
}

done_testing;


use strict;
use Test::More;

use CUDA::DeviceAPI;
use CUDA::DeviceAPI::Array;
use File::Spec;
use Cwd;

my $max = 10;
my $path = File::Spec->catfile(cwd(), qw/ t ptx 01_simple.ptx /);
my $host_data = pack('f*', 1..$max);

my $ctx = CUDA::DeviceAPI->new();

my $dev_ptr_1 = $ctx->malloc($host_data);
my $dev_ptr_2 = $ctx->malloc($host_data);
my $dev_ptr_3 = $ctx->malloc($host_data);

$ctx->transfer_h2d($host_data, $dev_ptr_1);
$ctx->transfer_h2d($host_data, $dev_ptr_2);

$ctx->run($path, 'kernel_sum', [
    $dev_ptr_1 => 'p',
    $dev_ptr_2 => 'p',
    $dev_ptr_3 => 'p',
    $max       => 'i',
], [
    $max
]
);

my $result = pack('f*', (0) x $max);
$ctx->transfer_d2h($dev_ptr_3, \$result);

my @results = unpack('f*', $result);

is_deeply(\@results, [map { $_ * 2 } 1..$max]);

done_testing;


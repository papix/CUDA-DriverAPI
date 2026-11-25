use v5.40;
use Test2::V0;
use FindBin;
use File::Spec;

use CUDA::DriverAPI;

skip_all "CUDA device is not available: $@" unless eval { CUDA::DriverAPI->new };

my $max       = 10;
my $max_data  = pack('i',  $max);
my $host_data = pack('f*', 1 .. $max);
my $path      = File::Spec->catfile($FindBin::Bin, qw/ ptx 01_simple.ptx /);

my $ctx = CUDA::DriverAPI->new;

my @dev_ptrs = map { $ctx->malloc($host_data) } 1 .. 3;
my $int_ptr  = $ctx->malloc($max_data);

$ctx->transfer_h2d($host_data, $dev_ptrs[0]);
$ctx->transfer_h2d($host_data, $dev_ptrs[1]);
$ctx->transfer_h2d($max_data,  $int_ptr);

subtest 'Run successfully' => sub {
    $ctx->run($path, 'kernel_sum', [ @dev_ptrs, $int_ptr ], [$max]);

    my $result = pack('f*', (0) x $max);
    $ctx->transfer_d2h($dev_ptrs[2], \$result);

    is [ unpack('f*', $result) ], [ map { $_ * 2 } 1 .. $max ];
};

subtest 'Run again with the cached library and function' => sub {
    $ctx->run($path, 'kernel_sum', [ @dev_ptrs, $int_ptr ], [$max]);

    is [ unpack('f*', $ctx->transfer_d2h($dev_ptrs[2])) ],          [ map { $_ * 2 } 1 .. $max ];
    is [ keys %{ $ctx->{libraries} } ],                             [$path];
    is [ keys %{ $ctx->{functions}{ $ctx->{libraries}{$path} } } ], ['kernel_sum'];
};

done_testing;

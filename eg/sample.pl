use strict;
use warnings;

use CUDA::DeviceAPI;
use CUDA::DeviceAPI::Array;
use Data::Dumper;

my $LENGTH = 4;
my $block_size = 32;

my $matrix = [
    [ 1, 2, 3, 4 ],
    [ 1, 2, 3, 4 ],
    [ 1, 2, 3, 4 ],
    [ 1, 2, 3, 4 ],
];

my $path = 'sample.ptx';
my $host_data = CUDA::DeviceAPI::Array->new($matrix);

my $ctx = CUDA::DeviceAPI->new();

my $dev_ptr_1 = $ctx->malloc_from($host_data => 'f');
my $dev_ptr_2 = $ctx->malloc_from($host_data => 'f');
my $dev_ptr_3 = $ctx->malloc($host_data->size() => 'f');

my $size = $ctx->ceil($LENGTH / $block_size);

$ctx->run($path, 'kernel', [
    $dev_ptr_1 => 'p',
    $dev_ptr_2 => 'p',
    $dev_ptr_3 => 'p',
    $LENGTH    => 'i',
], [
    $size, $size, 1, $block_size, $block_size, 1
]
);

my $result = $ctx->return($dev_ptr_3);

print Dumper $result;

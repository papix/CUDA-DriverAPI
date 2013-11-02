use strict;
use Test::More;

use CUDA::DeviceAPI;
use CUDA::DeviceAPI::Array;
use File::Spec;
use Cwd;
use Math::Matrix;

my $LENGTH = 4;
my $max = $LENGTH ** 2;
my $block_size = 32;

my $matrix = create_matrix($LENGTH, $LENGTH);

my $path = File::Spec->catfile(cwd(), qw/ t ptx 02_array_mul.ptx /);
my $host_data = CUDA::DeviceAPI::Array->new($matrix);

my $ctx = CUDA::DeviceAPI->new();
$ctx->init();

my $dev_ptr_1 = $ctx->malloc_from($host_data => 'f');
my $dev_ptr_2 = $ctx->malloc_from($host_data => 'f');
my $dev_ptr_3 = $ctx->malloc($host_data->size() => 'f');

my $size = $ctx->ceil($LENGTH / $block_size);

$ctx->run($path, 'kernel_sum', [
    $dev_ptr_1 => 'p',
    $dev_ptr_2 => 'p',
    $dev_ptr_3 => 'p',
    $LENGTH    => 'i',
], [
    $size, $size, 1, $block_size, $block_size, 1
]
);

my $result = $ctx->return($dev_ptr_3);

is_deeply($result, [
    [10, 10, 10, 10],
    [20, 20, 20, 20],
    [30, 30, 30, 30],
    [40, 40, 40, 40],
]);

sub create_matrix {
    my ($length, $width) = @_;

    my $i = 1;
    my $array = [];
    for my $j (0..$length - 1) {
        for my $k (0..$width - 1) {
            $array->[$j][$k] = $i;
        }
        $i++;
    }
    return $array;
}

done_testing;


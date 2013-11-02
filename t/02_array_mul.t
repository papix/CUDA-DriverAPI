use strict;
use Test::More;

use CUDA::DeviceAPI;
use CUDA::DeviceAPI::Array;
use File::Spec;
use Cwd;
use Math::Matrix;

my $length = 16;
my $max = $length ** 2;
my $block_size = 32;

my $matrix = matrix($length, $length);
my $matrix_array = matrix2array($matrix);

my $path = File::Spec->catfile(cwd(), qw/ t ptx 02_array_mul.ptx /);
my $host_data = CUDA::DeviceAPI::Array->new($matrix_array);

my @answer = @{ matrix2array($matrix * $matrix) };

my $ctx = CUDA::DeviceAPI->new();
$ctx->init();

my $dev_ptr_1 = $ctx->malloc_from($host_data => 'f');
my $dev_ptr_2 = $ctx->malloc_from($host_data => 'f');
my $dev_ptr_3 = $ctx->malloc($host_data->size() => 'f');

my $size = $ctx->ceil($length / $block_size);

$ctx->run($path, 'kernel_sum', [
    $dev_ptr_1 => 'p',
    $dev_ptr_2 => 'p',
    $dev_ptr_3 => 'p',
    $length    => 'i',
], [
    $size, $size, 1, $block_size, $block_size, 1
]
);

$ctx->transfer_d2h($dev_ptr_3, \my $result);

my @results = unpack 'f*', $result;

is_deeply(\@results, \@answer);

sub matrix {
    my ($length, $width) = @_;

    my $i = 1;
    my $array = [];
    for my $j (0..$length - 1) {
        for my $k (0..$width - 1) {
            $array->[$j][$k] = $i;
        }
        $i++;
    }
    return Math::Matrix->new(@{$array});
}

sub matrix2array {
    my ($matrix) = @_;

    $matrix =~ s/^\s+//;
    $matrix =~ s/\s+$//;

    return [ map { int($_) } split /\s+/, $matrix ];
}

done_testing;


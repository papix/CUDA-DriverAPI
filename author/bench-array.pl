#!/usr/bin/env perl
# Times host <-> device transfer of a nested Perl array, pure-Perl pack/unpack
# against Buffer->write_array/read_array.
#
#   perl -Mblib author/bench-array.pl [size] [repeats]
use v5.40;
use Time::HiRes qw(time);

use CUDA::DriverAPI;

my $n       = shift // 1024;
my $repeats = shift // 10;

my $ctx    = CUDA::DriverAPI->new;
my $matrix = [
    map {
        my $r = $_;
        [ map { $r + $_ / $n } 1 .. $n ]
    } 1 .. $n
];
my $buffer = $ctx->buffer(size => 4 * $n * $n);
my $into   = [];

my @cases = (
    [
        'pack + write' => sub {
            $buffer->write(join '', map { pack 'f*', @$_ } @$matrix);
        }
    ],
    [ 'write_array' => sub { $buffer->write_array($matrix, type => 'f32', rank => 2) } ],
    [
        'read + unpack' => sub {
            my @f = unpack 'f*', $buffer->read;
            [ map { [ splice @f, 0, $n ] } 1 .. $n ];
        }
    ],
    [ 'read_array'         => sub { $buffer->read_array(type => 'f32', dims => [ $n, $n ]) } ],
    [ 'read_array(into)'   => sub { $buffer->read_array(type => 'f32', dims => [ $n, $n ], into => $into) } ],
    [ 'fill(0)'            => sub { $buffer->fill(0); $ctx->synchronize } ],
    [ 'write("\0" x size)' => sub { $buffer->write("\0" x (4 * $n * $n)) } ],
);

printf "%dx%d f32, median of %d runs\n", $n, $n, $repeats;
for my $case (@cases) {
    my ($name, $code) = @$case;
    $code->();    # warm up
    my @times = sort { $a <=> $b } map { my $t = time; $code->(); time - $t } 1 .. $repeats;
    printf "  %-20s %8.2f ms\n", $name, 1000 * $times[ $#times / 2 ];
}

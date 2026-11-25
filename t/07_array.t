use v5.40;
use Test2::V0;

use CUDA::DriverAPI;

skip_all "CUDA device is not available: $@" unless eval { CUDA::DriverAPI->new };

my $ctx = CUDA::DriverAPI->new;

subtest 'round trip of rank 1, 2 and 3' => sub {
    my $vector = [ 1 .. 5 ];
    my $matrix = [
        map {
            my $r = $_;
            [ map { $r * 10 + $_ } 0 .. 3 ]
        } 0 .. 2
    ];
    my $cube = [
        map {
            my $i = $_;
            [
                map {
                    my $j = $_;
                    [ map { $i * 100 + $j * 10 + $_ } 0 .. 1 ]
                } 0 .. 2
            ]
        } 0 .. 3
    ];

    for my $case ([ $vector, 1, [5] ], [ $matrix, 2, [ 3, 4 ] ], [ $cube, 3, [ 4, 3, 2 ] ]) {
        my ($array, $rank, $dims) = @$case;
        my $buffer = $ctx->buffer(size => 4 * 24);

        is $buffer->write_array($array, type => 'i32', rank => $rank), $dims,  "rank $rank dims";
        is $buffer->read_array(type => 'i32', dims => $dims),          $array, "rank $rank values";
    }
};

subtest 'element types' => sub {
    my %cases = (
        i8  => [ -128,                     127 ],
        u8  => [  0,                       255 ],
        i16 => [ -32768,                   32767 ],
        u16 => [  0,                       65535 ],
        i32 => [ -2**31,                   2**31 - 1 ],
        u32 => [ 0,                        2**32 - 1 ],
        i64 => [ -9223372036854775807 - 1, 9223372036854775807 ],
        u64 => [  0,                       ~0 ],
        f32 => [ -1.5,                      0.25 ],
        f64 => [  0.1,                     -1e300 ],
    );
    for my $type (sort keys %cases) {
        my $buffer = $ctx->buffer(size => 16);
        $buffer->write_array($cases{$type}, type => $type);
        is $buffer->read_array(type => $type, dims => [2]), $cases{$type}, $type;
    }

    my $buffer = $ctx->buffer(size => 16);
    $buffer->write_array([ 1.5, -2.5 ], type => 'f32');
    is $buffer->read(size => 8), pack('f*', 1.5, -2.5), 'same bytes as pack';
};

subtest 'undef counts as 0, strings are numified' => sub {
    my $buffer = $ctx->buffer(size => 16);
    $buffer->write_array([ undef, '2', 3.0, undef ], type => 'i32');
    is $buffer->read_array(type => 'i32', dims => [4]), [ 0, 2, 3, 0 ];
};

subtest 'out-of-range floats become infinities, like pack' => sub {

    # pack 'f' turns anything beyond FLT_MAX into infinity, even values a
    # C cast would round down to FLT_MAX.
    my @values = (1e300, -1e300, 3.4028235e38, -3.4028235e38, 3.4028234e38);
    my $buffer = $ctx->buffer(size => 4 * @values);
    $buffer->write_array(\@values, type => 'f32');
    is $buffer->read, pack('f*', @values);
};

{
    # A tied scalar whose FETCH runs arbitrary code.
    package Tied::Hook;
    sub TIESCALAR ($class, $value, $hook) { return bless { value => $value, hook => $hook }, $class }
    sub FETCH     ($self)                 { $self->{hook}->(); return $self->{value} }
}

subtest 'tied elements are read once, and may not change the array' => sub {
    my $buffer = $ctx->buffer(size => 12);
    my $reads  = 0;
    my @row    = (1, undef, 3);
    tie $row[1], 'Tied::Hook', 2, sub { $reads++ };
    $buffer->write_array(\@row, type => 'i32');
    is $buffer->read_array(type => 'i32', dims => [3]), [ 1, 2, 3 ];
    is $reads, 1, 'FETCH ran once';

    my @shrinking = (1, undef, 3);
    tie $shrinking[1], 'Tied::Hook', 2, sub { splice @shrinking, 1 };
    like dies { $buffer->write_array(\@shrinking, type => 'i32') }, qr/Row  changed while it was being converted/;

    my @strings = ('1', 'abc', '3');
    {
        use warnings;
        local $SIG{__WARN__} = sub { @strings = () };
        like dies { $buffer->write_array(\@strings, type => 'i32') }, qr/Row  changed while it was being converted/,
            'a __WARN__ handler emptying the row';
    }

    my $square = $ctx->buffer(size => 16);
    my @rows   = ([ 1, undef ], [ 3, 4 ]);
    tie $rows[0][1], 'Tied::Hook', 2, sub { @rows = () };
    like dies { $square->write_array(\@rows, type => 'i32', rank => 2) }, qr/Row  changed while it was being converted/,
        'the parent array emptied by a nested element';
};

subtest 'invalid input' => sub {
    my $buffer = $ctx->buffer(size => 64);

    like dies { $buffer->write_array([ [ 1, 2 ], [3] ], type => 'f32', rank => 2) },
        qr/Row \[1\] has 1 elements, expected 2/;
    like dies { $buffer->write_array([ [ [1], [2] ], [ [3], [ 4, 5 ] ] ], type => 'f32', rank => 3) },
        qr/Row \[1\]\[1\] has 2 elements, expected 1/;
    like dies { $buffer->write_array([ [ 1, 2 ], 3 ], type => 'f32', rank => 2) },
        qr/Element \[1\] is not an array reference/;
    like dies { $buffer->write_array([ 1, 2 ], type => 'f32', rank => 2) }, qr/Element \[0\] is not an array reference/;
    like dies { $buffer->write_array([ [ 1, [2] ] ], type => 'f32', rank => 2) },
        qr/Element \[0\]\[1\] is a reference, not a number/;
    like dies { $buffer->write_array({}, type => 'f32') },             qr/Expected an array reference/;
    like dies { $buffer->write_array([1], type => 'f16') },            qr/Unknown array element type: f16/;
    like dies { $buffer->write_array([1]) },                           qr/type is required/;
    like dies { $buffer->write_array([1], type => 'f32', rank => 4) }, qr/Invalid rank: 4/;
    like dies { $buffer->write_array([ (0) x 17 ], type => 'f32') },
        qr/Array of 68 bytes exceeds the buffer of 64 bytes/;

    like dies { $buffer->read_array(type => 'f32', dims => [17]) },
        qr/Array of 68 bytes exceeds the buffer of 64 bytes/;
    like dies { $buffer->read_array(type => 'f32', dims => [ 4, 4, 1, 1 ]) },
        qr/dims must be an array reference of 1 to 3 sizes/;
    like dies { $buffer->read_array(type => 'f32', dims => [-1]) },            qr/Invalid dimension: -1/;
    like dies { $buffer->read_array(type => 'f32') },                          qr/dims is required/;
    like dies { $buffer->read_array(type => 'x', dims => [1]) },               qr/Unknown array element type: x/;
    like dies { $buffer->read_array(type => 'f32', dims => [1], into => {}) }, qr/into must be an array reference/;
};

subtest 'empty arrays' => sub {
    my $buffer = $ctx->buffer(size => 4);
    is $buffer->write_array([], type => 'f32', rank => 2), [ 0, 0 ];
    is $buffer->read_array(type => 'f32', dims => [ 0, 3 ]), [];
    is $buffer->read_array(type => 'f32', dims => [ 2, 0 ]), [ [], [] ];
};

subtest 'read_array into an existing array' => sub {
    my $buffer = $ctx->buffer(size => 4 * 6);
    $buffer->write_array([ [ 1, 2, 3 ], [ 4, 5, 6 ] ], type => 'f32', rank => 2);

    my $row0   = [ 9, 9, 9, 9 ];
    my $target = [ $row0, undef, ['extra'] ];
    my $result = $buffer->read_array(type => 'f32', dims => [ 2, 3 ], into => $target);

    ref_is $result,      $target, 'returns the given reference';
    ref_is $target->[0], $row0,   'existing rows are reused';
    is $target, [ [ 1, 2, 3 ], [ 4, 5, 6 ] ], 'resized to the shape';

    like dies { $buffer->read_array(type => 'f32', dims => [ 2, 3 ], into => [ 1, 2 ]) },
        qr/Element \[0\] is not an array reference/;

    # A tied array sees the rows it gets, through STORE.
    require Tie::Array;
    tie my @tied, 'Tie::StdArray';
    $buffer->read_array(type => 'f32', dims => [ 2, 3 ], into => \@tied);
    is [@tied], [ [ 1, 2, 3 ], [ 4, 5, 6 ] ], 'tied target';
};

subtest 'works with kernels' => sub {
    my $program = $ctx->compile(<<'CUDA');
extern "C" __global__ void twice(float *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= 2;
}
CUDA
    my $buffer = $ctx->buffer(size => 4 * 6);
    my $dims   = $buffer->write_array([ [ 1, 2, 3 ], [ 4, 5, 6 ] ], type => 'f32', rank => 2);
    $ctx->run($program, 'twice', [ $buffer, [ i32 => 6 ] ], [ 1, 1, 1, 32 ]);
    is $buffer->read_array(type => 'f32', dims => $dims), [ [ 2, 4, 6 ], [ 8, 10, 12 ] ];
};

done_testing;

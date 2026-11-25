use v5.40;
use Test2::V0;

use CUDA::DriverAPI;

skip_all "CUDA device is not available: $@" unless eval { CUDA::DriverAPI->new };
skip_all "PDL is not installed"             unless eval { require PDL::Lite; 1 };

my $ctx = CUDA::DriverAPI->new;

# The same rectangular data as a nested array (outermost dimension first) and
# as an ndarray (dimension 0 innermost).
my $nested = [ [ 1, 2, 3 ], [ 4, 5, 6 ] ];
my $pdl    = PDL->pdl(PDL::float(), $nested);

subtest 'an ndarray and a nested array write the same bytes' => sub {
    my $from_pdl    = $ctx->buffer(size => 24);
    my $from_nested = $ctx->buffer(size => 24);

    is $from_pdl->write_array($pdl), [ 2, 3 ], 'dims outermost first, type from the ndarray';
    $from_nested->write_array($nested, type => 'f32', rank => 2);
    is $from_pdl->read, $from_nested->read;

    is $from_pdl->read_array(type => 'f32', dims => [ 2, 3 ]), $nested, 'read back as a nested array';
};

subtest 'read as a new ndarray' => sub {
    my $buffer = $ctx->buffer(size => 24);
    $buffer->write_array($nested, type => 'f32', rank => 2);

    my $result = $buffer->read_array(type => 'f32', dims => [ 2, 3 ], as => 'pdl');
    isa_ok $result, 'PDL';
    is [ $result->dims ],  [ 3, 2 ], 'ndarray dims are the reverse';
    is "" . $result->type, 'float';
    is $result->unpdl,     $nested;
};

subtest 'read into an existing ndarray' => sub {
    my $buffer = $ctx->buffer(size => 24);
    $buffer->write_array($pdl);

    my $target = PDL->zeroes(PDL::float(), 6);
    my $result = $buffer->read_array(type => 'f32', dims => [ 2, 3 ], into => $target);
    ref_is $result, $target, 'returns the given ndarray';
    is [ $target->dims ], [ 3, 2 ], 'reshaped to the requested dims';
    is $target->unpdl, $nested;

    like dies { $buffer->read_array(type => 'f32', dims => [ 2, 3 ], into => PDL->zeroes(PDL::double(), 6)) },
        qr/into holds double, expected float for f32/;
    like dies { $buffer->read_array(type => 'f32', dims => [ 2, 3 ], into => [], as => 'pdl') },
        qr/into must be an ndarray when reading as 'pdl'/;
};

subtest 'element types' => sub {
    my %values = (
        i8  => [ -128,                     127 ],
        u8  => [  0,                       255 ],
        i16 => [ -32768,                   32767 ],
        u16 => [  0,                       65535 ],
        i32 => [ -2147483648,              2147483647 ],
        u32 => [  0,                       4294967295 ],
        i64 => [ -9223372036854775807 - 1, 9223372036854775807 ],
        f32 => [ -1.5,                     0.25 ],
        f64 => [  0.1,                     -1e300 ],
    );
    for my $type (sort keys %values) {
        my $buffer = $ctx->buffer(size => 16);
        $buffer->write_array($values{$type}, type => $type);
        my $result = $buffer->read_array(type => $type, dims => [2], as => 'pdl');
        is $result->unpdl, $values{$type}, $type;

        my $again = $ctx->buffer(size => 16);
        $again->write_array($result);
        is $again->read, $buffer->read, "$type ndarray writes the same bytes";
    }
};

subtest 'read into a view updates its parent' => sub {
    my $buffer = $ctx->buffer(size => 8);
    $buffer->write_array([ 1, 2 ], type => 'f32');

    my $parent = PDL->zeroes(PDL::float(), 4);
    $buffer->read_array(type => 'f32', dims => [2], into => $parent->slice('1:2'));
    is $parent->unpdl, [ 0, 1, 2, 0 ];
};

subtest 'an oversized read leaves into untouched' => sub {
    my $buffer = $ctx->buffer(size => 8);
    my $target = PDL->sequence(PDL::float(), 2);

    like dies { $buffer->read_array(type => 'f32', dims => [ 1000, 1000 ], into => $target) },
        qr/Array of 4000000 bytes exceeds the buffer of 8 bytes/;
    is [ $target->dims ], [2],      'not reshaped';
    is $target->unpdl,    [ 0, 1 ], 'not overwritten';
};

subtest 'type conversion and views' => sub {
    my $buffer = $ctx->buffer(size => 24);

    is $buffer->write_array(PDL->sequence(PDL::double(), 3, 2), type => 'f32'), [ 2, 3 ], 'converted to f32';
    is $buffer->read_array(type => 'f32', dims => [ 2, 3 ]), [ [ 0, 1, 2 ], [ 3, 4, 5 ] ];

    my $big  = PDL->sequence(PDL::float(), 4, 4);
    my $view = $big->slice('1:2,0:2');    # not contiguous in $big
    is $buffer->write_array($view),                          [ 3, 2 ];
    is $buffer->read_array(type => 'f32', dims => [ 3, 2 ]), [ [ 1, 2 ], [ 5, 6 ], [ 9, 10 ] ];
    is $big->unpdl->[0],                                     [ 0, 1, 2, 3 ], 'the parent is untouched';
};

subtest 'invalid input' => sub {
    my $buffer = $ctx->buffer(size => 8);

    like dies { $buffer->write_array(PDL->zeroes(PDL::float(), 3)) },
        qr/Array of 12 bytes exceeds the buffer of 8 bytes/;
    like dies { $buffer->write_array(PDL->pdl(PDL::float(), 1)) }, qr/Invalid rank: 0/;
    like dies { $buffer->write_array(PDL->zeroes(PDL::float(), 1, 1, 1, 1)) }, qr/Invalid rank: 4/;
    like dies { $buffer->write_array(PDL->zeroes(PDL::float(), 2), rank => 2) },
        qr/rank 2 does not match the ndarray's 1 dimensions/;
    like dies { $buffer->write_array(PDL->zeroes(PDL::float(), 2), type => 'f16') },
        qr/Unknown array element type: f16/;
    like dies { $buffer->write_array(PDL->zeroes(PDL::cdouble(), 1)) }, qr/Unsupported ndarray type: cdouble/;
    like dies { $buffer->read_array(type => 'f32', dims => [3], as => 'pdl') },
        qr/Array of 12 bytes exceeds the buffer of 8 bytes/;
    like dies { $buffer->read_array(type => 'f32', dims => [1], as => 'hash') }, qr/as must be 'array' or 'pdl'/;
    like dies { $buffer->read_array(type => 'f32', dims => [1], as => 'array', into => PDL->zeroes(PDL::float(), 1)) },
        qr/into is an ndarray, which as => 'array' cannot fill/;
};

subtest 'works with kernels' => sub {
    my $program = $ctx->compile(<<'CUDA');
extern "C" __global__ void twice(float *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= 2;
}
CUDA
    my $buffer = $ctx->buffer(size => 24);
    my $dims   = $buffer->write_array($pdl);
    $ctx->run($program, 'twice', [ $buffer, [ i32 => 6 ] ], [ 1, 1, 1, 32 ]);
    is $buffer->read_array(type => 'f32', dims => $dims, as => 'pdl')->unpdl, [ [ 2, 4, 6 ], [ 8, 10, 12 ] ];
};

done_testing;

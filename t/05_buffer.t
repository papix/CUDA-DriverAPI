use v5.40;
use Test2::V0;
use FindBin;
use File::Spec;

use CUDA::DriverAPI;

skip_all "CUDA device is not available: $@" unless eval { CUDA::DriverAPI->new };

my $path = File::Spec->catfile($FindBin::Bin, qw/ ptx 03_scalar.ptx /);
my $ctx  = CUDA::DriverAPI->new;

subtest 'create from data or size' => sub {
    my $data = $ctx->buffer(data => pack('l*', 1 .. 4));
    is $data->size,                   16;
    is [ unpack('l*', $data->read) ], [ 1 .. 4 ];

    my $sized = $ctx->buffer(size => 32);
    is $sized->size,                       32;
    is $ctx->{allocations}{ $sized->ptr }, 32;

    like dies { $ctx->buffer }, qr/Specify exactly one of data or size/;
    like dies { $ctx->buffer(data => 'x', size => 1) }, qr/Specify exactly one of data or size/;
    like dies { $ctx->buffer(size => '1.5') },          qr/Invalid buffer size: 1.5/;
    like dies { $ctx->buffer(size => -1) },             qr/Invalid buffer size: -1/;
};

subtest 'empty buffers' => sub {
    for my $buffer ($ctx->buffer(size => 0), $ctx->buffer(data => '')) {
        is $buffer->size, 0;
        is $buffer->read, '', 'read';
        ok lives { $buffer->write('') }, 'write';
        ok lives { $buffer->fill(7) },   'fill';
        is $buffer->write_array([], type => 'f32'), [0], 'write_array';
        is $buffer->read_array(type => 'f32', dims => [0]), [], 'read_array';
        is $buffer->read_array(type => 'f32', dims => [ 2, 0 ]), [ [], [] ];
        like dies { $buffer->write('x') },                     qr/outside the buffer of 0 bytes/;
        like dies { $buffer->read(size => 1) },                qr/outside the buffer of 0 bytes/;
        like dies { $buffer->write_array([1], type => 'u8') }, qr/exceeds the buffer of 0 bytes/;
    }

    my $empty = $ctx->buffer(size => 0);
    my $ptr   = $empty->ptr;
    ok $empty->free,                      'freed';
    ok !exists $ctx->{allocations}{$ptr}, 'and forgotten';
};

subtest 'read and write with offsets' => sub {
    my $buffer = $ctx->buffer(size => 8);

    is $buffer->write('abcdefgh'), $buffer, 'write returns the buffer';
    $buffer->write('XY', offset => 3);
    is $buffer->read, 'abcXYfgh';
    is $buffer->read(offset => 6),            'gh';
    is $buffer->read(offset => 2, size => 3), 'cXY';
    is $buffer->read(offset => 8),            '';

    like dies { $buffer->write('123', offset => 6) }, qr/Range 6\.\.9 is outside the buffer of 8 bytes/;
    like dies { $buffer->read(size   => 9) },  qr/Range 0\.\.9 is outside the buffer of 8 bytes/;
    like dies { $buffer->read(offset => 9) },  qr/Range 9\.\.9 is outside the buffer of 8 bytes/;
    like dies { $buffer->read(offset => -1) }, qr/Invalid offset or size: -1/;
};

subtest 'freed when it goes out of scope' => sub {
    my $ptr;
    {
        my $buffer = $ctx->buffer(size => 64);
        $ptr = $buffer->ptr;
        ok exists $ctx->{allocations}{$ptr};
    }
    ok !exists $ctx->{allocations}{$ptr};

    my $buffer = $ctx->buffer(size => 64);
    ok $buffer->free;
    ok !$buffer->free, 'second free is a no-op';
    like dies { $buffer->read }, qr/The buffer has already been freed/;
};

subtest 'passed as kernel arguments' => sub {
    my $n = 300;
    my $x = $ctx->buffer(data => pack('f*', 1 .. $n));
    my $y = $ctx->buffer(data => pack('f*', (2) x $n));

    $ctx->run($path, 'kernel_axpy', [ [ f32 => 3 ], $x, $y, [ i32 => $n ] ], [ int(($n + 127) / 128), 1, 1, 128 ]);
    is [ unpack('f*', $y->read) ], [ map { 3 * $_ + 2 } 1 .. $n ];
};

subtest 'buffers keep the context alive' => sub {
    my $buffer = do {
        my $scoped = CUDA::DriverAPI->new;
        $scoped->buffer(data => 'still here');
    };
    is $buffer->read, 'still here';
};

subtest 'destroy releases buffers that are still alive' => sub {
    my $scoped = CUDA::DriverAPI->new;
    my $buffer = $scoped->buffer(size => 16);

    ok $scoped->destroy;
    ok lives { $buffer->free }, 'freeing afterwards is a no-op';
};

subtest 'released buffers are refused as kernel arguments' => sub {
    my $freed = $ctx->buffer(size => 4);
    $freed->free;
    like dies { $ctx->run($path, 'kernel_axpy', [ $freed, $freed, $freed, [ i32 => 1 ] ]) },
        qr/The buffer has already been freed/;

    my $scoped    = CUDA::DriverAPI->new;
    my $destroyed = $scoped->buffer(size => 4);
    my @args      = ($destroyed, $destroyed, $destroyed, [ i32 => 1 ]);
    $scoped->destroy;
    $scoped->init;
    like dies { $scoped->run($path, 'kernel_axpy', \@args) }, qr/The buffer has already been freed/,
        'released by destroy';

    my $other = CUDA::DriverAPI->new->buffer(size => 4);
    like dies { $ctx->run($path, 'kernel_axpy', [ $other, $other, $other, [ i32 => 1 ] ]) },
        qr/The buffer belongs to another CUDA::DriverAPI object/;
};

subtest 'a stale buffer never touches a newer allocation at the same address' => sub {
    my $scoped = CUDA::DriverAPI->new;
    my $stale  = $scoped->buffer(data => 'old!');

    # Release the memory behind the buffer's back, then allocate until the
    # driver hands out the same address again.
    $scoped->free($stale->ptr);
    my @fresh;
    for (1 .. 64) {
        push @fresh, $scoped->buffer(data => 'new!');
        last if $fresh[-1]->ptr == $stale->ptr;
    }
    my ($reused) = grep { $_->ptr == $stale->ptr } @fresh;
    skip_all 'the driver did not reuse the address' unless $reused;

    like dies { $stale->read },       qr/The buffer has already been freed/, 'stale reads are refused';
    like dies { $stale->write('x') }, qr/The buffer has already been freed/, 'stale writes are refused';
    undef $stale;    # DESTROY must not free the reused allocation
    is $reused->read, 'new!', 'the newer buffer is intact';
    ok exists $scoped->{allocations}{ $reused->ptr };
};

subtest 'a stale program never unloads a newer one' => sub {
    my $scoped = CUDA::DriverAPI->new;
    my $source = qq{extern "C" __global__ void k() {}\n};
    my $stale  = $scoped->compile($source);

    $scoped->destroy;
    $scoped->init;
    my $fresh = $scoped->compile($source);
    like dies { $scoped->run($stale, 'k', []) }, qr/The program has already been released/;
    undef $stale;
    ok lives { $scoped->run($fresh, 'k', []); $scoped->synchronize }, 'the newer program still runs';
};

done_testing;

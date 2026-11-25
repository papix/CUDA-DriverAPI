use v5.40;
use Test2::V0;
use FindBin;
use File::Spec;

use CUDA::DriverAPI;

skip_all "CUDA device is not available: $@" unless eval { CUDA::DriverAPI->new };

my $path = File::Spec->catfile($FindBin::Bin, qw/ ptx 03_scalar.ptx /);
my $n    = 1000;
my $ctx  = CUDA::DriverAPI->new;

subtest 'malloc allocates the length of a byte string' => sub {
    is $ctx->{allocations}{ $ctx->malloc(pack('f*', (0) x $n)) }, 4 * $n;
    is $ctx->{allocations}{ $ctx->malloc(4096) }, 4, 'a number is data too';

    like dies { $ctx->malloc('') }, qr/Invalid allocation size: 0/;
};

subtest 'scalar kernel arguments and asynchronous launches' => sub {
    my $x = $ctx->malloc(pack('f*', 1 .. $n));
    my $y = $ctx->malloc(pack('f*', (1) x $n));
    $ctx->transfer_h2d(pack('f*', 1 .. $n),  $x);
    $ctx->transfer_h2d(pack('f*', (1) x $n), $y);

    # y = 2x + y, three times; launches queue up without waiting.
    my @config = (int(($n + 255) / 256), 1, 1, 256);
    $ctx->run($path, 'kernel_axpy', [ [ f32 => 2 ], $x, $y, [ i32 => $n ] ], \@config) for 1 .. 3;
    $ctx->synchronize;

    is [ unpack('f*', $ctx->transfer_d2h($y)) ], [ map { 6 * $_ + 1 } 1 .. $n ];
    is length($ctx->transfer_d2h($y, 8)),        8,  'partial read';
    is $ctx->transfer_d2h($y, 0),                '', 'empty read';
};

subtest 'invalid kernel arguments' => sub {
    my $message = qr/must be a device pointer, a buffer or \[type => value\]/;

    like dies { $ctx->run($path, 'kernel_axpy', [ [ f16 => 1 ] ]) }, qr/Unknown kernel argument type: f16/;
    like dies { $ctx->run($path, 'kernel_axpy', [ [ i32 => [1] ] ]) },
        qr/Kernel argument \[i32 => \.\.\.\] must be a number, not a reference/;
    my $freed = $ctx->buffer(size => 4);
    $freed->free;
    like dies { $ctx->run($path, 'kernel_axpy', [ [ ptr => $freed ] ]) }, qr/The buffer has already been freed/,
        '[ptr => $buffer] is checked like a plain buffer';
    like dies { $ctx->run($path, 'kernel_axpy', [ ['f32'] ]) }, $message;
    like dies { $ctx->run($path, 'kernel_axpy', [ {} ]) },      $message;
};

subtest 'transfers are bounds-checked against the allocation' => sub {
    my $ptr = $ctx->malloc('x' x 16);

    like dies { $ctx->transfer_h2d('x' x 17, $ptr) }, qr/Transfer of 17 bytes exceeds the allocation of 16 bytes/;
    like dies { my $buf = "\0" x 17; $ctx->transfer_d2h($ptr, \$buf) }, qr/Transfer of 17 bytes exceeds/;
    like dies { $ctx->transfer_d2h($ptr, 17) },                         qr/Transfer of 17 bytes exceeds/;
    like dies { $ctx->transfer_d2h($ptr + 4) },    qr/Size is required for a pointer not allocated by malloc/;
    like dies { $ctx->transfer_d2h($ptr, -1) },    qr/Invalid size: -1/;
    like dies { $ctx->transfer_d2h($ptr, 'abc') }, qr/Invalid size: abc/;

    ok lives { $ctx->transfer_h2d('x' x 16, $ptr) };
    is $ctx->transfer_d2h($ptr + 4, 4), 'xxxx', 'offset pointers work with an explicit size';
};

subtest 'wide strings are rejected' => sub {
    like dies { $ctx->malloc("\x{3042}") }, qr/Wide character/;
};

done_testing;

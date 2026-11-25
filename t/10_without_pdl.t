use v5.40;

# Hide PDL even where it is installed, to check that it stays optional.
BEGIN {
    unshift @INC, sub ($hook, $file) {
        die "Can't locate $file (hidden by the test)\n" if $file =~ m{\APDL(?:/|\.pm\z)};
        return;
    };
}

use Test2::V0;

use CUDA::DriverAPI;

skip_all "CUDA device is not available: $@" unless eval { CUDA::DriverAPI->new };

my $ctx    = CUDA::DriverAPI->new;
my $buffer = $ctx->buffer(size => 24);

ok !grep({ m{\APDL} } keys %INC), 'loading CUDA::DriverAPI does not load PDL';

is $buffer->write_array([ [ 1, 2, 3 ], [ 4, 5, 6 ] ], type => 'f32', rank => 2), [ 2, 3 ];
is $buffer->read_array(type => 'f32', dims => [ 2, 3 ]), [ [ 1, 2, 3 ], [ 4, 5, 6 ] ], 'arrays work without PDL';

like dies { $buffer->read_array(type => 'f32', dims => [ 2, 3 ], as => 'pdl') }, qr/PDL is required for as => 'pdl'/;
ok !grep({ m{\APDL} } keys %INC), 'still not loaded';

done_testing;

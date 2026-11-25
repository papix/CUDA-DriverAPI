use v5.40;
use Test2::V0;
use Config;
use Scalar::Util qw(blessed);
use FindBin;
use File::Spec;

use CUDA::DriverAPI;

skip_all "CUDA device is not available: $@" unless eval { CUDA::DriverAPI->new };

my $path = File::Spec->catfile($FindBin::Bin, qw/ ptx 01_simple.ptx /);

my @warnings;
local $SIG{__WARN__} = sub { push @warnings, @_ };

subtest 'free' => sub {
    my $ctx = CUDA::DriverAPI->new;
    my $ptr = $ctx->malloc(pack('f*', 1 .. 4));

    ok $ctx->free($ptr);
    like dies { $ctx->free($ptr) }, qr/Not exist: $ptr/;
};

subtest 'errors from the driver carry the CUDA error name' => sub {
    my $ctx = CUDA::DriverAPI->new;

    like dies { $ctx->run($path, 'no_such_kernel', []) }, qr/CUDA error at LibraryGetKernel: CUDA_ERROR_NOT_FOUND/;
    like dies { $ctx->run(File::Spec->catfile($FindBin::Bin, 'no_such.ptx'), 'kernel_sum', []) },
        qr/CUDA error at LibraryLoadFromFile: CUDA_ERROR_FILE_NOT_FOUND/;
    like dies { $ctx->run($path, 'kernel_sum', [], [ (1) x 8 ]) }, qr/Too many launch config values/;
};

subtest 'destroy' => sub {
    my $ctx = CUDA::DriverAPI->new;
    $ctx->malloc('x' x 16);

    ok $ctx->destroy;
    ok !$ctx->destroy, 'second destroy is a no-op';
    like dies { $ctx->malloc('x' x 16) }, qr/CUDA context has already been destroyed/;

    ok $ctx->init, 'init after destroy retains the context again';
    ok $ctx->free($ctx->malloc('x' x 16));
};

subtest 'invalid device' => sub {
    like dies { CUDA::DriverAPI->new(device => 9999) }, qr/CUDA error at DeviceGet: CUDA_ERROR_INVALID_DEVICE/;
};

subtest 'objects are not cloned into ithreads' => sub {
    skip_all 'perl is built without ithreads' unless $Config{useithreads};
    require threads;

    my $ctx = CUDA::DriverAPI->new;
    my $ptr = $ctx->malloc('x' x 16);

    # CLONE_SKIP turns the clone into an unblessed placeholder, so no DESTROY runs there.
    my $seen = threads->create(sub { blessed($ctx) // 'unblessed' })->join;

    is $seen, 'unblessed';
    ok $ctx->free($ptr), 'the parent still owns its allocation';
};

is \@warnings, [], 'no warnings, including from DESTROY';

done_testing;

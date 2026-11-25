use v5.40;
use Test2::V0;

my @warnings;
local $SIG{__WARN__} = sub { push @warnings, @_ };

ok lives { require CUDA::DriverAPI }, 'CUDA::DriverAPI loads, with or without an NVIDIA driver';
is \@warnings, [], 'without warnings';

# Without a usable driver, new() reports why instead of failing to load.
my $error = eval { CUDA::DriverAPI->new; 1 } ? undef : $@;
if (defined $error) {
    like $error, qr/\A(?:CUDA driver (?:not found|not usable|too old)|CUDA error at \w+:)/,
        "new() explains the missing driver or device: $error";
}
else {
    pass 'a CUDA device is available';
}

done_testing;

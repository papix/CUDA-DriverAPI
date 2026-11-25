use v5.40;
use Test2::V0;

my @warnings;
local $SIG{__WARN__} = sub { push @warnings, @_ };

ok lives { require CUDA::DriverAPI }, 'CUDA::DriverAPI loads';
is \@warnings, [], 'without warnings';

done_testing;

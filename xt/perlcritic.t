use v5.40;
use FindBin;
use Test::Perl::Critic -profile => "$FindBin::Bin/perlcriticrc";

all_critic_ok('lib', 'builder');

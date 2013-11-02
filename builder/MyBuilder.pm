package builder::MyBuilder;
use strict;
use warnings FATAL => 'all';
use 5.008008;
use base 'Module::Build::XSUtil';

sub new {
    my ( $self, %args ) = @_;
    $self->SUPER::new(
        %args,
        extra_linker_flags => '-lcuda',
        cc_warnings => 0,
        config => { cc => 'gcc', ld => 'gcc' },
    );
}

1;
__END__

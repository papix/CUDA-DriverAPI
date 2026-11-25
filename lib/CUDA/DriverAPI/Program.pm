package CUDA::DriverAPI::Program;
use v5.40;

our $VERSION = "0.02";

# Created by CUDA::DriverAPI->compile. Like a buffer, it keeps its context
# object alive and unloads its code when it goes away.
sub _new ($class, $ctx, $library, $image, $log, $token) {
    return bless { ctx => $ctx, library => $library, image => $image, log => $log, token => $token }, $class;
}

sub log   ($self) { return $self->{log} }
sub image ($self) { return $self->{image} }

sub DESTROY ($self) {
    my $ctx = delete $self->{ctx} or return;
    $ctx->_release_library($self->{library}, $self->{token});
    return;
}

sub CLONE_SKIP { return 1 }

1;

__END__

=encoding utf-8

=for stopwords NVRTC PTX cubin

=head1 NAME

CUDA::DriverAPI::Program - CUDA code compiled at run time

=head1 SYNOPSIS

    my $program = $ctx->compile($cuda_source, options => [ '-std=c++20' ]);
    warn $program->log if length $program->log;
    $ctx->run($program, 'kernel_name', \@args, \@config);

=head1 DESCRIPTION

Code compiled with NVRTC by L<CUDA::DriverAPI/compile> and loaded on the
device. Pass it to L<CUDA::DriverAPI/run>. The code is unloaded when the
object goes out of scope.

=head1 METHODS

=over 4

=item C<< $program->log >>

The compiler log, such as warnings. Empty when there is nothing to report.

=item C<< $program->image >>

The compiled cubin, or PTX when compiled for a virtual architecture. It can be
saved to a file and passed to L<CUDA::DriverAPI/run> later.

=back

=cut

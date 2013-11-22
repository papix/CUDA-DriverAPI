package CUDA::DriverAPI;
use 5.008005;
use strict;
use warnings;
use Carp;

our $VERSION = "0.01";

use XSLoader;
XSLoader::load(__PACKAGE__, $VERSION);

sub new {
    my ($class, %argv) = @_;

    bless {
        context => CUDA::DriverAPI::_init(),
        addr    => {},
    }, $class;
}

sub init {
    my ($self) = @_;
    $self->{context} ||= CUDA::DriverAPI::_init();
}

sub malloc {
    my ($self, $size) = @_;

    my $addr = CUDA::DriverAPI::_malloc($self->{context}, $size);
    $self->{addr}->{$addr} = 1;

    return $addr;
}

sub transfer_h2d {
    my ($self, $src_var, $dst_ptr) = @_;
    CUDA::DriverAPI::_transfer_h2d($self->{context}, $src_var, $dst_ptr);
}

sub transfer_d2h {
    my ($self, $src_ptr, $dst_var) = @_;
    CUDA::DriverAPI::_transfer_d2h($self->{context}, $src_ptr, ${$dst_var});
}

sub run {
    my ($self, $ptx_path, $function, $args, $config) = @_;
    CUDA::DriverAPI::_run($self->{context}, $ptx_path, $function, $args, $config);
}

sub free {
    my ($self, $addr) = @_;

    Carp::croak("Not exist: $addr") unless exists $self->{addr}->{$addr};
    if ($self->{addr}->{$addr}) {
        CUDA::DriverAPI::_free($self->{context}, $addr);
        delete $self->{addr}->{$addr};
        return 1;
    } else {
        return 0;
    }
}

sub destroy {
    my ($self) = @_;

    return 0 unless $self->{context};

    for my $addr (keys %{$self->{addr}}) {
        if (exists $self->{addr}->{$addr}) {
            $self->free($addr);
            delete $self->{addr}->{$addr};
        }
    }

    CUDA::DriverAPI::_destroy($self->{context});
    delete $self->{context};
    return 1;
}

sub DESTROY {
    my ($self) = @_;
    $self->destroy;
}

1;

__END__

=encoding utf-8

=head1 NAME

CUDA::DriverAPI - It's new $module

=head1 SYNOPSIS

    use CUDA::DriverAPI;

=head1 DESCRIPTION

CUDA::DriverAPI is ...

=head1 LICENSE

Copyright (C) papix.

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

=head1 AUTHOR

papix E<lt>mail@papix.netE<gt>

=cut


package CUDA::DeviceAPI;
use 5.008005;
use strict;
use warnings;
use Carp;

our $VERSION = "0.01";

use XSLoader;
XSLoader::load(__PACKAGE__, $VERSION);

my %VARIABLE_TYPE = (
    'p' => 0,
    'i' => 1,
    'f' => 2,
    'd' => 3,
);

sub new {
    my ($class, %argv) = @_;

    bless {
        context => CUDA::DeviceAPI::_init(),
        addr    => {},
    }, $class;
}

sub init {
    my ($self) = @_;
    $self->{context} ||= CUDA::DeviceAPI::_init();
}

sub malloc {
    my ($self, $size) = @_;

    my $addr = CUDA::DeviceAPI::_malloc($self->{context}, $size);
    $self->{addr}->{$addr} = 1;

    return $addr;
}

sub transfer_h2d {
    my ($self, $src_var, $dst_ptr) = @_;
    CUDA::DeviceAPI::_transfer_h2d($self->{context}, $src_var, $dst_ptr);
}

sub transfer_d2h {
    my ($self, $src_ptr, $dst_var) = @_;
    CUDA::DeviceAPI::_transfer_d2h($self->{context}, $src_ptr, ${$dst_var});
}

sub run {
    my ($self, $ptx_path, $function, $args, $config) = @_;

    Carp::croak("Error!") if @{$args} % 2 != 0;
    for my $i (0 .. $#{$args}) {
        if ($i % 2 == 1) {
            $args->[$i] = $VARIABLE_TYPE{$args->[$i]};
        }
    }

    CUDA::DeviceAPI::_run($self->{context}, $ptx_path, $function, $args, $config);
}

sub free {
    my ($self, $addr) = @_;

    Carp::croak("Not exist: $addr") unless exists $self->{addr}->{$addr};
    if ($self->{addr}->{$addr}) {
        CUDA::DeviceAPI::_free($self->{context}, $addr);
        delete $self->{addr}->{$addr};
        return 1;
    } else {
        return 0;
    }
}

sub destroy {
    my ($self) = @_;

    if ($self->{context}) {
        for my $addr (keys %{$self->{addr}}) {
            if (exists $self->{addr}->{$addr}) {
                $self->free($addr);
                delete $self->{addr}->{$addr};
            }
        }

        CUDA::DeviceAPI::_destroy($self->{context});
        delete $self->{context};
    }
}

sub DESTROY {
    my ($self) = @_;
    $self->destroy;
}

1;

__END__

=encoding utf-8

=head1 NAME

CUDA::DeviceAPI - It's new $module

=head1 SYNOPSIS

    use CUDA::DeviceAPI;

=head1 DESCRIPTION

CUDA::DeviceAPI is ...

=head1 LICENSE

Copyright (C) papix.

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

=head1 AUTHOR

papix E<lt>mail@papix.netE<gt>

=cut


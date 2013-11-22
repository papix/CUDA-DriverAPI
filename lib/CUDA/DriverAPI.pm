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

CUDA::DriverAPI - CUDA bindings for Perl using CUDA Driver API

=head1 SYNOPSIS

    use CUDA::DriverAPI;
    use File::Spec;

    my $ctx = CUDA::DriverAPI->new();
    my $path = File::Spec->catfile(qw/ path to ptx file /);
    
    my $max = 10;
    my $max_b  = pack('i', $max);
    my $host_b = pack('f*', 1 .. $max);

    my $ptr1 = $ctx->malloc($host_b);
    my $ptr2 = $ctx->malloc($host_b);
    my $ptr3 = $ctx->malloc($host_b);
    my $ptr4 = $ctx->malloc($max_b);

    $ctx->transfer_h2d($host_b, $ptr1);
    $ctx->transfer_h2d($host_b, $ptr2);
    $ctx->transfer_h2d($max_b, $ptr4);

    $ctx->run($path, [ $ptr1, $ptr2, $ptr3, $ptr4 ], [ $max ]);

    my $result = $host_b;
    $ctx->transfer_d2h($ptr3, \$result);

=head1 DESCRIPTION

CUDA::DriverAPI is CUDA bindings module for Perl.

=head1 DEPENDENCIES

This module requiers CUDA 4.0.

=head1 LICENSE

Copyright (C) papix.

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

=head1 AUTHOR

papix E<lt>mail@papix.netE<gt>

=cut


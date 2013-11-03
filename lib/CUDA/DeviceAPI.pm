package CUDA::DeviceAPI;
use 5.008005;
use strict;
use warnings;
use Carp;
use CUDA::DeviceAPI::Array;
use CUDA::DeviceAPI::Util;

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
        ptr     => {},
    }, $class;
}

sub init {
    my ($self) = @_;

    $self->{context} ||= CUDA::DeviceAPI::_init();
}

sub run {
    my ($self, $ptx_path, $function, $args, $config) = @_;

    Carp::croak("Error!") if @{$args} % 2 != 0;
    for my $i (0 .. $#{$args}) {
        if ($i % 2) {
            $args->[$i] = $VARIABLE_TYPE{$args->[$i]};
        } else {
            $args->[$i] = ref $args->[$i] eq 'HASH'
                ? $args->[$i]->{addr} : $args->[$i];
        }
    }

    CUDA::DeviceAPI::_run($self->{context}, $ptx_path, $function, $args, $config);
}

sub array {
    my $self = shift;
    return CUDA::DeviceAPI::Array->new(@_);
}

sub transfer_h2d {
    my ($self, $array, $type, $dst_ptr) = @_;

    my $type_symbol = type_symbol($type);
    $array = ref $array eq 'CUDA::DeviceAPI::Array' ? $array : CUDA::DeviceAPI::Array->new($array);
    my $data = ref2array($array);

    my $binary_data = pack("$type_symbol*", @{$data});
    CUDA::DeviceAPI::_transfer_h2d($self->{context}, $binary_data, $dst_ptr);
}

sub transfer_d2h {
    my ($self, $src_ptr, $dst_var) = @_;

    my $type_symbol = $src_ptr->{type};
    ${$dst_var} = pack("$type_symbol*", (0) x $src_ptr->{elem});

    CUDA::DeviceAPI::_transfer_d2h($self->{context}, $src_ptr->{addr}, ${$dst_var});
}

sub malloc_from {
    my ($self, $array, $type) = @_;

    my $type_symbol = type_symbol($type);
    $array = ref $array eq 'CUDA::DeviceAPI::Array' ? $array : CUDA::DeviceAPI::Array->new($array);

    my $addr = $self->malloc($array->size, $type_symbol);
    $self->transfer_h2d($array, $type_symbol, $addr->{addr});

    return $addr;
}

sub malloc {
    my ($self, $size, $type) = @_;

    my $type_symbol = type_symbol($type);
    $size = ref $size eq 'CUDA::DeviceAPI::Array' ? $size->size : $size;

    my $elem = size2elem($size);
    my $binary_data = pack("$type_symbol*", (0) x $elem);
    my $addr = CUDA::DeviceAPI::_malloc($self->{context}, $elem * bit_length($type_symbol));

    $self->{addr}->{$addr} = 1;

    return +{
        size => $size,
        elem => $elem,
        type => $type_symbol,
        addr => $addr,
    };
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
        for my $addr (grep { $self->{addr}->{$_} == 1 } keys %{$self->{addr}}) {
            $self->free($addr);
        }

        CUDA::DeviceAPI::_destroy($self->{context});
        delete $self->{context};
    }
}

sub DESTROY {
    my ($self) = @_;
    $self->destroy;
}

sub ceil {
    my ($self, $n) = @_;
    return int($n) == $n ? $n : int($n) + 1;
}

sub return {
    my ($self, $src_ptr) = @_;
    $self->transfer_d2h($src_ptr, \my $dst_val);

    return array2ref($dst_val, $src_ptr->{type}, $src_ptr->{size});
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


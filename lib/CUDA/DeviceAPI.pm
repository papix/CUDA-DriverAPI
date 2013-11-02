package CUDA::DeviceAPI;
use 5.008005;
use strict;
use warnings;
use Carp;
use CUDA::DeviceAPI::Array;

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
        context => undef,
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

sub transfer_h2d {
    my ($self, $array, $type, $dst_ptr) = @_;

    my $type_symbol = $self->_type_symbol($type);
    $array = ref $array eq 'CUDA::DeviceAPI::Array' ? $array : CUDA::DeviceAPI::Array->new($array);
    my $data = $self->_array_normalize($array);

    my $binary_data = pack("$type_symbol*", @{$data});
    CUDA::DeviceAPI::_transfer_h2d($self->{context}, $binary_data, $dst_ptr);
}

sub transfer_d2h {
    my ($self, $src_ptr, $dst_var) = @_;

    my $type_symbol = $src_ptr->{type};
    ${$dst_var} = pack("$type_symbol*", (0) x $src_ptr->{size});

    CUDA::DeviceAPI::_transfer_d2h($self->{context}, $src_ptr->{addr}, ${$dst_var});
}

sub size {
    my ($self, $array) = @_;
    my ($x_size, $y_size, $z_size) = (scalar @{$array}, 1, 1);

    if (ref $array->[0] eq 'ARRAY') {
        for my $y (@{$array}) {
            $y_size = $y_size < scalar @{$y} ? scalar @{$y} : $y_size;
            if (ref $array->[0]->[0] eq 'ARRAY') {
                for my $z (@{$y}) {
                    $z_size = $z_size < scalar @{$z} ? scalar @{$z} : $z_size;
                }
            }
        }
    }

    return $x_size * $y_size * $z_size;
}

sub malloc_from {
    my ($self, $array, $type) = @_;

    my $type_symbol = $self->_type_symbol($type);
    $array = ref $array eq 'CUDA::DeviceAPI::Array' ? $array : CUDA::DeviceAPI::Array->new($array);

    my $addr = $self->malloc($array->size, $type_symbol);
    $self->transfer_h2d($array, $type_symbol, $addr->{addr});

    return $addr;
}

sub _array_normalize {
    my ($self, $array) = @_;

    my @data;

    for my $i (0..$array->size('x') - 1) {
        if ($array->dim == 1) {
            push @data, $array->{data}->[$i] || 0;
        } else {
            for my $j (0..$array->size('y') - 1) {
                if ($array->dim == 2) {
                    push @data, $array->{data}->[$i]->[$j] || 0;
                } else {
                    for my $k (0..$array->size('z') - 1) {
                        push @data, $array->{data}->[$i]->[$j]->[$k] || 0;
                    }
                }
            }
        }
    }

    return \@data;
}

sub malloc {
    my ($self, $size, $type) = @_;

    my $type_symbol = $self->_type_symbol($type);
    $size = ref $size eq 'CUDA::DeviceAPI::Array' ? $size->size : $size;

    my $binary_data = pack("$type_symbol*", (0) x $size);
    my $addr = CUDA::DeviceAPI::_malloc($self->{context}, $size * $self->_bit_length($type_symbol));

    $self->{addr}->{$addr} = 1;

    return {
        size => $size,
        type => $type_symbol,
        addr => $addr,
    };
}

sub _bit_length {
    my ($self, $type) = @_;

    if ($type eq 'f') {
        return 4;
    } else {
        Carp::croak("ERROR");
    }
}

sub _type_symbol {
    my ($self, $type) = @_;

    if ($type =~ /f(loat)?/i) {
        return 'f';
    } elsif ($type =~ /d(ouble)?/i) {
        return 'd';
    } else {
        Carp::croak("ERROR");
    }
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

sub ceil {
    my ($self, $n) = @_;
    my $int_n = int($n);

    if ($int_n == $n) {
        return $n;
    } else {
        return int($n) + 1;
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


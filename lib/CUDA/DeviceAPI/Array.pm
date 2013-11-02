package CUDA::DeviceAPI::Array;
use strict;
use warnings;
use Carp;

sub new {
    my ($class, $data, $opts) = @_;

    Carp::croak("ERROR") if ref $data ne 'ARRAY';

    my ($dim, $elem, $size) = _size($data);

    bless {
        data => $data,
        size => $size,
        elem => $elem,
        dim  => $dim,
    }, $class;
}

sub data {
    my ($self) = @_;
    return $self->{data};
}

sub size {
    my ($self, $dim) = @_;

    unless (defined $dim) {
        return $self->{size};
    } else {
        return $self->{size}->{$dim} || 0;
    }
}

sub elem {
    my ($self) = @_;
    return $self->{elem};
}

sub dim {
    my ($self) = @_;
    return $self->{dim};
}

sub _size {
    my ($data) = @_;
    my ($x_size, $y_size, $z_size) = (scalar @{$data}, 0, 0);
    my ($dim, $elem) = (1, $x_size);

    if (ref $data->[0] eq 'ARRAY') {
        for my $y (@{$data}) {
            $y_size = $y_size < scalar @{$y} ? scalar @{$y} : $y_size;
            if (ref $data->[0]->[0] eq 'ARRAY') {
                for my $z (@{$y}) {
                    $z_size = $z_size < scalar @{$z} ? scalar @{$z} : $z_size;
                }
            }
        }
    }

    if ($y_size > 0) {
        $dim++; $elem *= $y_size;
        if ($z_size > 0) {
            $dim++; $elem *= $z_size;
        }
    }

    return ($dim, $elem, { x => $x_size, y => $y_size, z => $z_size });
}

1;


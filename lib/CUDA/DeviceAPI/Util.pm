package CUDA::DeviceAPI::Util;
use strict;
use warnings;

use parent 'Exporter';

our @EXPORT = qw/ ref2array array2ref bit_length type_symbol size2elem /;

sub ref2array {
    my ($array) = @_;
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

sub array2ref {
    my ($array_str, $type, $size) = @_;

    my $type_symbol = type_symbol($type);
    my @array = unpack "$type_symbol*", $array_str;
    my $array_ref = [];

    for my $i (0..$size->{x} - 1) {
        if ($size->{y} == 1) {
            my $n = shift @array;
            $array_ref->[$i] = $n;
        } else {
            for my $j (0..$size->{y} - 1) {
                if ($size->{z} == 1) {
                    my $n = shift @array;
                    $array_ref->[$i]->[$j] = $n;
                } else {
                    for my $k (0..$size->{z} - 1) {
                        my $n = shift @array;
                        $array_ref->[$i]->[$j]->[$k] = $n;
                    }
                }
            }
        }
    }

    return $array_ref;
}

sub bit_length {
    my ($type) = @_;

    $type = type_symbol($type);
    if ($type eq 'f') {
        return 4;
    } else {
        Carp::croak("ERROR");
    }
}

sub type_symbol {
    my ($type) = @_;

    if ($type =~ /f(loat)?/i) {
        return 'f';
    } elsif ($type =~ /d(ouble)?/i) {
        return 'd';
    } else {
        Carp::croak("ERROR");
    }
}

sub size2elem {
    my ($size) = @_;

    my $elem = $size->{x};
    $elem *= $size->{y} > 0 ? $size->{y} : 1;
    $elem *= $size->{z} > 0 ? $size->{z} : 1;

    return $elem;
}

1;

package CUDA::DriverAPI::Buffer;
use v5.40;
use Carp         ();
use Scalar::Util qw(blessed);

our $VERSION = "0.02";

# Created by CUDA::DriverAPI->buffer. Holds a reference to its context object
# so that the context outlives every buffer allocated from it, and the token
# that tells its allocation apart from a later one at the same address.
sub _new ($class, $ctx, $ptr, $size, $token) {
    return bless { ctx => $ctx, ptr => $ptr, size => $size, token => $token }, $class;
}

sub ptr  ($self) { return $self->{ptr} }
sub size ($self) { return $self->{size} }

sub write ($self, $data, %args) {
    my $offset = $args{offset} // 0;
    $self->_check_range($offset, CUDA::DriverAPI::_byte_length($data));
    $self->_ctx->transfer_h2d($data, $self->{ptr} + $offset);
    return $self;
}

sub read ($self, %args) {
    my $offset = $args{offset} // 0;
    my $size   = $args{size}   // ($offset =~ /\A[0-9]+\z/ && $offset < $self->{size} ? $self->{size} - $offset : 0);
    $self->_check_range($offset, $size);
    return $self->_ctx->transfer_d2h($self->{ptr} + $offset, $size);
}

sub write_array ($self, $array, %args) {
    return $self->_write_pdl($array, %args) if _is_pdl($array);

    my $type = $args{type} // Carp::croak("type is required");
    my $rank = $args{rank} // 1;
    Carp::croak("Invalid rank: $rank (expected 1 to 3)") unless $rank =~ /\A[1-3]\z/;

    return CUDA::DriverAPI::_write_array($self->_ctx->_context, $self->{ptr}, $self->{size}, $array, $type, $rank);
}

sub read_array ($self, %args) {
    my $type = $args{type} // Carp::croak("type is required");
    my $dims = $args{dims} // Carp::croak("dims is required");
    my $as   = $args{as}   // (_is_pdl($args{into}) ? 'pdl' : 'array');
    Carp::croak("dims must be an array reference of 1 to 3 sizes")
        unless ref $dims eq 'ARRAY' && @$dims >= 1 && @$dims <= 3;
    for my $dim (@$dims) {
        Carp::croak("Invalid dimension: " . ($dim // 'undef')) unless defined $dim && $dim =~ /\A[0-9]+\z/;
    }
    Carp::croak("as must be 'array' or 'pdl'") unless $as eq 'array' || $as eq 'pdl';

    return $self->_read_pdl($type, $dims, $args{into}) if $as eq 'pdl';
    return CUDA::DriverAPI::_read_array($self->_ctx->_context, $self->{ptr}, $self->{size}, $type, $dims, $args{into});
}

sub fill ($self, $byte = 0) {
    Carp::croak("Invalid byte: $byte") unless $byte =~ /\A[0-9]+\z/ && $byte <= 255;
    CUDA::DriverAPI::_memset($self->_ctx->_context, $self->{ptr}, $byte, $self->{size});
    return $self;
}

sub free ($self) {
    my $ctx = delete $self->{ctx} or return false;
    $ctx->_release_allocation($self->{ptr}, $self->{token});
    return true;
}

sub DESTROY ($self) {
    $self->free;
    return;
}

sub CLONE_SKIP { return 1 }

# The context, as long as this buffer's allocation is still alive: it is gone
# after free, after $ctx->free($buffer->ptr), and after $ctx->destroy.
sub _ctx ($self) {
    my $ctx = $self->{ctx};
    Carp::croak("The buffer has already been freed") unless $ctx && $ctx->_owns($self->{ptr}, $self->{token});
    return $ctx;
}

# ---------------------------------------------------------------------------
# PDL support. PDL is optional: nothing here loads it unless the caller passes
# an ndarray or asks for one with `as => 'pdl'`. An ndarray stores its
# elements contiguously with dimension 0 varying fastest, which is exactly the
# row-major layout of write_array/read_array with the dimensions reversed, so
# the data is copied to and from the device without per-element conversion.
# ---------------------------------------------------------------------------

# Element types and the PDL type names that store them.
#<<< keep the pairs aligned
my %PDL_TYPE_OF = (
    i8  => 'sbyte',    u8  => 'byte',
    i16 => 'short',    u16 => 'ushort',
    i32 => 'long',     u32 => 'ulong',
    i64 => 'longlong', u64 => 'ulonglong',
    f32 => 'float',    f64 => 'double',
);
#>>>
my %ELEM_TYPE_OF = reverse %PDL_TYPE_OF;

sub _is_pdl ($value) {
    return blessed($value) && $value->isa('PDL');
}

sub _elem_type_of_pdl ($pdl) {
    my $name = "" . $pdl->type;
    $name = _indx_type() if $name eq 'indx';
    return $ELEM_TYPE_OF{$name} // Carp::croak("Unsupported ndarray type: " . $pdl->type);
}

# indx is a signed integer as wide as a pointer.
sub _indx_type () {
    return PDL::Core::howbig(PDL::indx()) == 8 ? 'longlong' : 'long';
}

sub _check_array_size ($self, $bytes) {
    Carp::croak("Array of $bytes bytes exceeds the buffer of $self->{size} bytes") if $bytes > $self->{size};
    return;
}

sub _write_pdl ($self, $pdl, %args) {
    my @dims = reverse $pdl->dims;
    Carp::croak("Invalid rank: " . @dims . " (expected 1 to 3)") unless @dims >= 1 && @dims <= 3;
    Carp::croak("rank $args{rank} does not match the ndarray's " . @dims . " dimensions")
        if defined $args{rank} && $args{rank} != @dims;

    my $type = $args{type}         // _elem_type_of_pdl($pdl);
    my $want = $PDL_TYPE_OF{$type} // Carp::croak("Unknown array element type: $type");
    $pdl = $pdl->convert(PDL::Type->new($want)) if _elem_type_of_pdl($pdl) ne $type;
    $pdl->make_physical;    # slices and other views get their own data

    # Hand the ndarray's own data string to XS: no copy on the host.
    my $data = $pdl->get_dataref;
    $self->_check_array_size(length $$data);
    CUDA::DriverAPI::_transfer_h2d($self->_ctx->_context, $$data, $self->{ptr}, $self->{size});
    return \@dims;
}

sub _read_pdl ($self, $type, $dims, $into) {
    my $pdl_type = $PDL_TYPE_OF{$type} // Carp::croak("Unknown array element type: $type");
    my @pdl_dims = reverse @$dims;
    my $pdl;

    if (defined $into) {
        Carp::croak("into must be an ndarray when reading as 'pdl'") unless _is_pdl($into);
        Carp::croak("into holds " . $into->type . ", expected $pdl_type for $type")
            unless _elem_type_of_pdl($into) eq $type;
        $into->reshape(@pdl_dims) unless "@{[ $into->dims ]}" eq "@pdl_dims";
        $pdl = $into;
    }
    else {
        eval { require PDL::Lite; 1 } or Carp::croak("PDL is required for as => 'pdl': $@");
        $pdl = PDL->new_from_specification(PDL::Type->new($pdl_type), @pdl_dims);
    }
    $pdl->make_physical;

    my $data = $pdl->get_dataref;
    $self->_check_array_size(length $$data);
    CUDA::DriverAPI::_transfer_d2h($self->_ctx->_context, $self->{ptr}, $$data, $self->{size});
    $pdl->upd_data;
    return $pdl;
}

sub _check_range ($self, $offset, $size) {
    for my $value ($offset, $size) {
        Carp::croak("Invalid offset or size: $value") unless $value =~ /\A[0-9]+\z/;
    }
    Carp::croak("Range $offset..@{[ $offset + $size ]} is outside the buffer of $self->{size} bytes")
        if $offset + $size > $self->{size};
    return;
}

1;

__END__

=encoding utf-8

=for stopwords ndarray ndarrays

=head1 NAME

CUDA::DriverAPI::Buffer - Device memory freed automatically

=head1 SYNOPSIS

    my $buffer = $ctx->buffer(size => 4 * 1024);
    $buffer->write(pack('f*', 1 .. 1024));
    my @values = unpack('f*', $buffer->read);

=head1 DESCRIPTION

A block of device memory allocated by L<CUDA::DriverAPI/buffer>. It is freed
when the object goes out of scope, and can be passed directly as a kernel
argument to L<CUDA::DriverAPI/run>.

=head1 METHODS

=over 4

=item C<< $buffer->ptr >>

The device pointer.

=item C<< $buffer->size >>

The size in bytes.

=item C<< $buffer->write($bytes, offset => $offset) >>

Copies C<$bytes> to the buffer, starting C<$offset> bytes in (default: 0).
Returns the buffer.

=item C<< $buffer->read(offset => $offset, size => $size) >>

Returns C<$size> bytes copied from the buffer, starting C<$offset> bytes in.
By default it reads everything from C<$offset> (default: 0) to the end.

=item C<< $buffer->write_array($array_ref, type => $type, rank => $rank) >>

Converts a rectangular nested array of depth C<$rank> (1 to 3, default 1) to
elements of C<$type> in row-major order and copies them to the start of the
buffer. The conversion runs in C, without building an intermediate string in
Perl. Returns the dimensions as an array reference, e.g. C<[ 1024, 1024 ]>.

C<$type> is one of C<i8>, C<u8>, C<i16>, C<u16>, C<i32>, C<u32>, C<i64>,
C<u64>, C<f32> and C<f64>. C<undef> elements count as 0. It croaks, naming the
index, on a row whose length differs from the first row's, on a row that is
not an array reference, and on a leaf that is a reference.

=item C<< $buffer->write_array($ndarray, type => $type, rank => $rank) >>

The same with a L<PDL> ndarray instead of a nested array. Its data is copied
as is, without per-element conversion. C<type> defaults to the ndarray's own
type (an ndarray of another type is converted first), C<rank> must match its
number of dimensions if given, and slices and other views work. The returned
dimensions are outermost first, i.e. the reverse of C<< $ndarray->dims >>.

=item C<< $buffer->read_array(type => $type, dims => \@dims) >>

=item C<< $buffer->read_array(type => $type, dims => \@dims, into => $array_ref) >>

Copies C<product(@dims)> elements of C<$type> from the start of the buffer
and returns them as a nested array of shape C<@dims> (1 to 3 sizes). With
C<into>, it fills the given array in place instead, reusing its rows and
scalars and resizing every level to C<@dims>, and returns C<$array_ref>.

=item C<< $buffer->read_array(type => $type, dims => \@dims, as => 'pdl') >>

=item C<< $buffer->read_array(type => $type, dims => \@dims, into => $ndarray) >>

The same, returning a new L<PDL> ndarray, or filling the given one in place
(it must already have the matching type and is reshaped to C<@dims>). The
data is copied without per-element conversion, and the ndarray's dimensions
are the reverse of C<@dims>.

=back

=head2 Dimensions and PDL

C<dims> always lists the outermost dimension first: a nested array
C<< $m->[$row][$col] >> has C<dims> C<[ $rows, $cols ]>. A PDL ndarray lists
its innermost dimension first, so the same data as an ndarray has
C<< $ndarray->dims >> C<($cols, $rows)>. Both store the elements in the same
order in memory, which is why ndarrays need no conversion.

PDL is optional. Nothing loads it unless an ndarray is passed in or
C<< as => 'pdl' >> is requested, and the methods behave the same with or
without it. Complex and long double ndarrays are not supported.

=over 4

=item C<< $buffer->fill($byte) >>

Sets every byte of the buffer to C<$byte> (0 to 255, default 0) and returns
the buffer. Like L<CUDA::DriverAPI/run>, it is asynchronous and ordered
before later kernels and reads.

=item C<< $buffer->free >>

Frees the memory now instead of when the object goes out of scope.

=back

Reads and writes croak when the range does not fit in the buffer.

=cut

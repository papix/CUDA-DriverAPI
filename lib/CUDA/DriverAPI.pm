package CUDA::DriverAPI;
use v5.40;
use Carp         ();
use Scalar::Util qw(blessed);

use CUDA::DriverAPI::Buffer;
use CUDA::DriverAPI::Program;

our $VERSION = "0.02";

use XSLoader;
XSLoader::load(__PACKAGE__, $VERSION);

# pack() formats of the scalar kernel argument types, in native byte order.
#<<< keep the pairs aligned
my %ARG_FORMAT = (
    i8  => 'c', u8  => 'C',
    i16 => 's', u16 => 'S',
    i32 => 'l', u32 => 'L',
    i64 => 'q', u64 => 'Q',
    f32 => 'f', f64 => 'd',
    ptr => 'Q',
);
#>>>

sub new ($class, %args) {
    my $self = bless {
        device      => $args{device} // 0,
        context     => undef,
        allocations => {},    # device pointer => size in bytes
        libraries   => {},    # file path, or the handle of a compiled program => CUlibrary
        functions   => {},    # CUlibrary => { name => CUfunction }
        owners      => {},    # device pointer or CUlibrary => token of its Buffer/Program
    }, $class;
    $self->init;
    return $self;
}

sub init ($self) {
    $self->{context} //= _init($self->{device});
    return true;
}

sub buffer ($self, %args) {
    my ($data, $size) = @args{qw(data size)};
    Carp::croak("Specify exactly one of data or size") unless defined($data) xor defined($size);

    my $ptr    = $self->_alloc($size // _byte_length($data));
    my $buffer = CUDA::DriverAPI::Buffer->_new($self, $ptr, $self->{allocations}{$ptr}, $self->_own($ptr));
    $buffer->write($data) if defined $data;
    return $buffer;
}

sub compile ($self, $source, %args) {
    my @options = @{ $args{options} // [] };
    push @options, $self->_default_arch
        unless grep { /\A(?:-arch|--gpu-architecture)\b/ } @options;

    # After the caller's options, so that their -I directories come first.
    push @options, map { "--include-path=$_" } _toolkit_include_dirs();

    my ($image, $log) = _compile($source, $args{name} // 'kernel.cu', \@options);
    return $self->_program($image, $log);
}

sub load ($self, $image) {
    return $self->_program($image, '');
}

sub compute_capability ($self) {
    return _compute_capability($self->_context, $self->{device});
}

sub malloc ($self, $data) {
    return $self->_alloc(_byte_length($data));
}

sub free ($self, $ptr) {
    Carp::croak("Not exist: $ptr") unless delete $self->{allocations}{$ptr};
    delete $self->{owners}{$ptr};
    _free($self->_context, $ptr);
    return true;
}

sub transfer_h2d ($self, $src, $dst_ptr) {
    _transfer_h2d($self->_context, $src, $dst_ptr, $self->_limit($dst_ptr));
    return;
}

sub transfer_d2h ($self, $src_ptr, $dst = undef) {
    if (ref $dst) {
        _transfer_d2h($self->_context, $src_ptr, $$dst, $self->_limit($src_ptr));
        return $$dst;
    }

    my $size = $dst // $self->{allocations}{$src_ptr}
        // Carp::croak("Size is required for a pointer not allocated by malloc: $src_ptr");
    Carp::croak("Invalid size: $size") unless $size =~ /\A[0-9]+\z/;
    Carp::croak("Transfer of $size bytes exceeds the allocation of " . $self->_limit($src_ptr) . " bytes")
        if $size > $self->_limit($src_ptr);
    return _read($self->_context, $src_ptr, $size);
}

sub run ($self, $code, $function, $args, $config = []) {
    my $library = $self->_resolve_library($code);
    my $func    = $self->{functions}{$library}{$function} //= _get_function($self->_context, $library, $function);
    _launch($self->_context, $func, [ map { $self->_pack_arg($_) } @$args ], $config);
    return;
}

sub synchronize ($self) {
    _synchronize($self->_context);
    return;
}

sub destroy ($self) {
    my $ctx = delete $self->{context} or return false;

    # Release everything even if a step fails (e.g. a sticky kernel error),
    # then report the first failure.
    my @errors;
    my $attempt = sub ($code, @args) {
        try {
            $code->(@args);
        }
        catch ($e) {
            push @errors, $e;
        }
        return;
    };
    $attempt->(\&_synchronize,    $ctx);
    $attempt->(\&_free,           $ctx, $_) for keys %{ $self->{allocations} };
    $attempt->(\&_unload_library, $ctx, $_) for values %{ $self->{libraries} };
    $attempt->(\&_destroy,        $self->{device});
    $self->@{qw(allocations libraries functions owners)} = ({}, {}, {}, {});

    die $errors[0] if @errors;
    return true;
}

sub DESTROY ($self) {
    $self->destroy;
    return;
}

# Driver handles must not be shared with cloned interpreters (ithreads),
# otherwise each clone would free them again in DESTROY.
sub CLONE_SKIP { return 1 }

sub _context ($self) {
    return $self->{context} // Carp::croak("CUDA context has already been destroyed");
}

sub _alloc ($self, $size) {
    Carp::croak("Invalid allocation size: $size") unless $size =~ /\A[1-9][0-9]*\z/;

    my $ptr = _malloc($self->_context, $size);
    $self->{allocations}{$ptr} = $size;
    return $ptr;
}

# Buffers and Programs identify their resource by a token as well as by its
# address: the driver reuses addresses, so after free or destroy a stale object
# could otherwise release or use a newer allocation at the same address.
my $last_token = 0;

sub _own ($self, $handle) {
    return $self->{owners}{$handle} = ++$last_token;
}

sub _owns ($self, $handle, $token) {
    return defined $token && ($self->{owners}{$handle} // 0) == $token;
}

# Called by Buffer and Program when they go away. A no-op once the context or
# the resource has already been released (e.g. by free or destroy).
sub _release_allocation ($self, $ptr, $token) {
    return unless $self->_owns($ptr, $token);
    delete $self->{owners}{$ptr};
    _free($self->{context}, $ptr) if $self->{context} && delete $self->{allocations}{$ptr};
    return;
}

sub _release_library ($self, $library, $token) {
    my $key = _program_key($library);
    return unless $self->_owns($key, $token);
    delete $self->{owners}{$key};
    delete $self->{libraries}{$key} // return;
    delete $self->{functions}{$library};
    _unload_library($self->{context}, $library) if $self->{context};
    return;
}

# The toolkit's header directories, found when the module was built and saved
# by builder/MyBuilder.pm in the generated CUDA::DriverAPI::ConfigData.
sub _toolkit_include_dirs () {
    state $dirs = do {
        my $saved =
            eval { require CUDA::DriverAPI::ConfigData; CUDA::DriverAPI::ConfigData->config('nvrtc_include_dirs') };
        [ grep { -d } @{ $saved // [] } ];
    };
    return @$dirs;
}

# Compiled programs are keyed apart from file paths in `libraries` and from
# device pointers in `owners`.
sub _program_key ($library) {
    return "program:$library";
}

sub _program ($self, $image, $log) {
    my $library = _load_library_data($self->_context, $image);
    my $key     = _program_key($library);
    $self->{libraries}{$key} = $library;
    return CUDA::DriverAPI::Program->_new($self, $library, $image, $log, $self->_own($key));
}

sub _resolve_library ($self, $code) {
    if (blessed $code) {
        Carp::croak("Not a compiled program: $code") unless $code->isa('CUDA::DriverAPI::Program');
        Carp::croak("The program belongs to another CUDA::DriverAPI object") unless $code->{ctx} == $self;
        Carp::croak("The program has already been released")
            unless $self->_owns(_program_key($code->{library}), $code->{token});
        return $code->{library};
    }
    return $self->{libraries}{$code} if exists $self->{libraries}{$code};
    return $self->{libraries}{$code} = _load_library($self->_context, $code);
}

# Compile for the device's own architecture (cubin, no JIT) when NVRTC supports
# it; otherwise emit PTX for the newest older architecture and let the driver
# JIT-compile it.
sub _default_arch ($self) {
    my ($major, $minor) = $self->compute_capability;
    my $device = $major * 10 + $minor;
    my @archs  = _nvrtc_supported_archs();

    return "--gpu-architecture=sm_$device" if grep { $_ == $device } @archs;
    my ($newest) = sort { $b <=> $a } grep { $_ < $device } @archs;
    Carp::croak("NVRTC supports no architecture for compute capability $major.$minor") unless $newest;
    return "--gpu-architecture=compute_$newest";
}

# Pointers that were not returned by malloc (e.g. offsets into a buffer) are
# not bounds-checked here; the driver still rejects out-of-range copies.
sub _limit ($self, $ptr) {
    return $self->{allocations}{$ptr} // ~0;
}

# A freed buffer, or one released by destroy, would launch the kernel on a
# dangling pointer: an asynchronous, sticky illegal-address error at best.
sub _buffer_ptr ($self, $buffer) {
    my $owner = $buffer->{ctx};
    Carp::croak("The buffer has already been freed")
        unless $owner && $owner->_owns($buffer->{ptr}, $buffer->{token});
    Carp::croak("The buffer belongs to another CUDA::DriverAPI object") unless $owner == $self;
    return $buffer->{ptr};
}

sub _pack_arg ($self, $arg) {
    return pack($ARG_FORMAT{ptr}, $self->_buffer_ptr($arg)) if blessed $arg && $arg->isa('CUDA::DriverAPI::Buffer');
    return pack($ARG_FORMAT{ptr}, $arg) unless ref $arg;

    Carp::croak("Kernel argument must be a device pointer, a buffer or [type => value]")
        unless ref $arg eq 'ARRAY' && @$arg == 2;
    my ($type, $value) = @$arg;
    my $format = $ARG_FORMAT{$type} // Carp::croak("Unknown kernel argument type: $type");
    return $self->_pack_arg($value) if $type eq 'ptr' && blessed $value && $value->isa('CUDA::DriverAPI::Buffer');

    # A reference would be packed as its Perl address, a garbage device pointer.
    Carp::croak("Kernel argument [$type => ...] must be a number, not a reference") if ref $value;
    return pack($format, $value);
}

1;

__END__

=encoding utf-8

=for stopwords cubin fatbin NVRTC JIT

=head1 NAME

CUDA::DriverAPI - CUDA bindings for Perl using CUDA Driver API

=head1 SYNOPSIS

    use CUDA::DriverAPI;

    my $ctx = CUDA::DriverAPI->new();

    my $program = $ctx->compile(<<'CUDA');
    extern "C" __global__ void axpy(float a, const float *x, float *y, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a * x[i] + y[i];
    }
    CUDA

    my $n = 1000;
    my $x = $ctx->buffer(data => pack('f*', 1 .. $n));
    my $y = $ctx->buffer(size => 4 * $n);
    $y->write(pack('f*', (1) x $n));

    $ctx->run($program, 'axpy', [ [ f32 => 2 ], $x, $y, [ i32 => $n ] ],
              [ int(($n + 255) / 256), 1, 1, 256 ]);

    my @result = unpack('f*', $y->read);    # 2 * x + 1

    # Buffers are freed when they go out of scope.

=head1 DESCRIPTION

CUDA::DriverAPI is CUDA bindings module for Perl.

It uses the device's primary context (shared with the CUDA Runtime API),
loads kernels with the context-independent C<cuLibrary*> API, and compiles
CUDA C++ source at run time with NVRTC.

=head1 METHODS

=over 4

=item C<< CUDA::DriverAPI->new(%args) >>

Retains the primary context of a device. C<device> is the device ordinal
(default: 0).

=item C<< $ctx->buffer(data => $bytes) >>

=item C<< $ctx->buffer(size => $size) >>

Allocates device memory and returns a L<CUDA::DriverAPI::Buffer>, which frees
the memory when it goes out of scope. With C<data>, the memory is
C<length($bytes)> bytes and initialized with C<$bytes>; with C<size>, it is
C<$size> bytes and uninitialized.

=item C<< $ctx->compile($source, %args) >>

Compiles CUDA C++ source with NVRTC and returns a L<CUDA::DriverAPI::Program>
to pass to C<run>. Kernels must be declared C<extern "C"> to be found by name.

C<name> is the file name shown in compiler messages (default: C<kernel.cu>).
C<options> is an array reference of NVRTC options such as C<-std=c++20>. Unless
it contains C<--gpu-architecture>, the code is compiled for the device's own
architecture. The headers of the CUDA Toolkit the module was built with
(e.g. C<cuda_fp16.h>, C<cooperative_groups.h>, C<< cuda/std/... >>) are found
without options; directories given with C<-I> are searched first. (For a
toolkit installed into F</usr> by distribution packages only F</usr/include/cccl>
is added, since F</usr/include> also holds host headers; pass C<-I/usr/include>
yourself if a kernel needs headers from there.) A compile error croaks with the
compiler log.

=item C<< $ctx->load($image) >>

Loads a compiled image (cubin, fatbin or PTX, e.g. a saved
C<< $program->image >>) and returns a L<CUDA::DriverAPI::Program>, like
C<compile> but without NVRTC. Use it to cache compiled code across processes,
keyed by C<compute_capability>. The driver loads code lazily, so an invalid
image is reported by the first C<run> that uses it.

=item C<< $ctx->compute_capability >>

Returns the device's compute capability as C<($major, $minor)>, e.g.
C<(8, 9)>.

=item C<< $ctx->run($code, $function, \@args, \@config) >>

Launches the kernel C<$function> in C<$code>, which is either a program from
C<compile> or the path of a PTX, cubin or fatbin file. Files and functions are
looked up once and cached.

Each element of C<@args> is a L<CUDA::DriverAPI::Buffer>, a device pointer, or
a C<< [ $type => $value ] >> pair passed by value, where C<$type> is one of
C<i8>, C<u8>, C<i16>, C<u16>, C<i32>, C<u32>, C<i64>, C<u64>, C<f32>, C<f64>
and C<ptr>. A buffer must come from this object and not be freed yet.

C<@config> is
C<(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes)>;
omitted values default to C<(1, 1, 1, 1, 1, 1, 0)>.

The launch is asynchronous: C<run> returns without waiting for the kernel.
Reading memory back, freeing it, C<destroy> and C<synchronize> wait for it, and
an error raised by the kernel is reported by the next of these calls.

=item C<< $ctx->synchronize >>

Waits until all launched kernels have finished.

=item C<< $ctx->destroy >>

Waits for pending work, frees all memory and loaded code, and releases the
primary context. Called automatically when the object and every buffer and
program created from it have gone out of scope.

=back

=head2 Raw device pointers

These methods work with plain device pointers instead of buffer objects.
Memory allocated by C<malloc> is not freed automatically before C<destroy>.

=over 4

=item C<< $ctx->malloc($bytes) >>

Allocates C<length($bytes)> bytes of device memory and returns the device
pointer. The content is not copied; use C<transfer_h2d>.

=item C<< $ctx->free($ptr) >>

Frees device memory allocated by C<malloc>.

=item C<< $ctx->transfer_h2d($bytes, $ptr) >>

Copies C<$bytes> to the device memory C<$ptr>.

=item C<< $ctx->transfer_d2h($ptr [, $size]) >>

Returns C<$size> bytes copied from the device memory C<$ptr>. C<$size> defaults
to the size of the allocation.

=item C<< $ctx->transfer_d2h($ptr, \$buffer) >>

Copies C<length($buffer)> bytes from the device memory C<$ptr> into C<$buffer>.

=back

Transfers into and out of memory allocated by C<malloc> croak when they would
run past the end of the allocation.

=head1 PERFORMANCE

=over 4

=item *

C<run>, C<fill> and the copies queued before them do not wait for the GPU.
Launch as many kernels as needed and read the results once at the end;
reading waits for everything launched before it.

=item *

Use C<write_array> and C<read_array> for nested Perl arrays instead of
C<pack>/C<unpack>: they convert in C and avoid an intermediate string. Pass
C<into> to C<read_array> in loops to reuse the same Perl array.

=item *

Use C<fill> instead of writing a string of zeros to initialize output buffers.

=item *

NVRTC takes a noticeable time on the first compile. Save C<< $program->image >>
to a file keyed by C<compute_capability> and C<load> it in later runs; see
F<eg/matmul.pl>.

=back

=head1 ERRORS

Every method croaks on failure. Errors from the driver read
C<CUDA error at E<lt>callE<gt>: E<lt>nameE<gt> (E<lt>codeE<gt>): E<lt>descriptionE<gt>>,
e.g. C<CUDA error at LibraryGetKernel: CUDA_ERROR_NOT_FOUND (500): named
symbol not found>, and NVRTC compile errors include the compiler log.

Because launches are asynchronous, an error raised while a kernel runs (such
as an illegal memory access) is reported by the next call that waits for the
GPU. Such errors are sticky: the device's primary context, shared by every
object for that device, stays unusable for the rest of the process.

=head1 THREADS

Objects are not cloned into new Perl ithreads; create a new object in each
thread instead.

=head1 INSTALLATION

Requirements: Perl 5.40 or later, an NVIDIA driver, and the CUDA Toolkit 12.0
or later including NVRTC (C<cuda.h>, C<nvrtc.h> and C<libnvrtc>).

    cpanm CUDA::DriverAPI

or from a checkout:

    cpanm --with-configure --installdeps .
    perl Build.PL
    ./Build
    ./Build test
    ./Build install

The toolkit is found without configuration in the usual places, in this
order: the directory of C<nvcc> on C<PATH>, F</usr/local/cuda>, the newest
C</usr/local/cuda-X.Y>, F</opt/cuda>, C<$CONDA_PREFIX>, and F</usr>
(distribution packages). C<perl Build.PL> prints the one it uses. To choose
another, set C<CUDA_HOME> (or C<CUDA_PATH>) to the directory that contains
C<include/cuda.h>; it is then the only place looked at.

On WSL, C<libcuda> is taken from F</usr/lib/wsl/lib>;
install only the toolkit (e.g. the C<cuda-toolkit-13-3> package from NVIDIA's
C<wsl-ubuntu> repository), not a Linux driver. Tests that need a GPU are
skipped when no CUDA device is available.

=head1 EXAMPLES

=over 4

=item F<eg/axpy.pl>

Compiles a kernel, moves data with C<write_array>/C<read_array> and launches it.

=item F<eg/matmul.pl>

Matrix multiplication with the compiled kernel cached on disk.

=item C<author/bench-array.pl>

Compares C<write_array>/C<read_array> with C<pack>/C<unpack>.

=back

=head1 SEE ALSO

L<CUDA::DriverAPI::Buffer>, L<CUDA::DriverAPI::Program>,
L<CUDA Driver API|https://docs.nvidia.com/cuda/cuda-driver-api/>,
L<NVRTC|https://docs.nvidia.com/cuda/nvrtc/>

=head1 LICENSE

Copyright (C) papix.

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

=head1 AUTHOR

papix E<lt>mail@papix.netE<gt>

=cut

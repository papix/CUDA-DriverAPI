package builder::MyBuilder;
use v5.40;
use parent 'Module::Build::XSUtil';
use Cwd            ();
use File::Basename qw(dirname);
use File::Spec;

# Finds the CUDA Toolkit that provides cuda.h, nvrtc.h and libnvrtc, without
# configuration where possible (see cuda_home_candidates). libcuda ships with
# the NVIDIA driver and is not linked: the XS loads it at run time, so the
# module builds and loads on machines without a driver.
sub new ($class, %args) {
    my $cuda_home = find_cuda_home(cuda_home_candidates());

    # A wrong explicit setting is a configuration error, not a missing platform.
    my ($explicit) = grep { length($ENV{$_} // '') } qw(CUDA_HOME CUDA_PATH);
    die "$explicit=$ENV{$explicit} does not contain include/cuda.h and include/nvrtc.h of a CUDA Toolkit.\n"
        if $explicit && !defined $cuda_home;

    # "OS unsupported" makes CPAN Testers report NA instead of FAIL on machines
    # without the toolkit.
    die "OS unsupported: the CUDA Toolkit 12.0 or later with NVRTC was not found "
        . "(looked for include/cuda.h and include/nvrtc.h in "
        . join(', ', cuda_home_candidates())
        . "). Install it, or set CUDA_HOME to its directory.\n"
        unless defined $cuda_home;

    # A toolkit installed into the system prefix (distribution packages) is on
    # the default include and library paths; adding them again only reorders
    # the system search order.
    my $system           = $cuda_home eq '/usr';
    my @include_dirs     = $system ? () : ("$cuda_home/include");
    my @toolkit_lib_dirs = $system ? () : grep { -d } ("$cuda_home/lib64", "$cuda_home/lib");

    my @search_flags = map { "-L$_" } @toolkit_lib_dirs;
    my @rpath_flags  = map { "-Wl,-rpath,$_" } @toolkit_lib_dirs;

    print "Using the CUDA Toolkit in $cuda_home\n";
    my $builder = $class->SUPER::new(
        %args,
        include_dirs       => [@include_dirs],
        extra_linker_flags => [ @search_flags, @rpath_flags, '-lnvrtc', '-ldl' ],
        cc_warnings        => 0,
    );

    # Saved in the generated CUDA::DriverAPI::ConfigData, so that compile()
    # can let NVRTC find the toolkit headers (cuda_fp16.h, cuda/std/..., ...).
    # For a system install, /usr/include also holds the host's C and C++
    # headers, which would shadow NVRTC's built-in ones; pass only cccl there.
    my $include            = "$cuda_home/include";
    my @nvrtc_include_dirs = grep { -d } ($system ? () : $include), "$include/cccl";
    $builder->config_data(cuda_home          => $cuda_home);
    $builder->config_data(nvrtc_include_dirs => \@nvrtc_include_dirs);
    return $builder;
}

# Where to look, in order. An explicit CUDA_HOME or CUDA_PATH is the only
# candidate, so a wrong setting fails loudly instead of silently picking
# another toolkit.
sub cuda_home_candidates () {
    for my $var (qw(CUDA_HOME CUDA_PATH)) {
        return ($ENV{$var}) if length($ENV{$var} // '');
    }

    my @candidates;
    if (my $nvcc = which('nvcc')) {

        # The toolkit nvcc resolves to (a symlink such as /usr/local/bin/nvcc or
        # an alternatives link), then the directory it was found in, for
        # Debian's /usr/bin/nvcc -> /usr/lib/nvidia-cuda-toolkit/bin/nvcc whose
        # headers are in /usr/include.
        push @candidates, dirname(dirname(Cwd::realpath($nvcc) // $nvcc)), dirname(dirname($nvcc));
    }
    push @candidates, '/usr/local/cuda';
    push @candidates, sort { version_of($b) <=> version_of($a) } glob('/usr/local/cuda-[0-9]*');
    push @candidates, '/opt/cuda';    # Arch Linux
    push @candidates, $ENV{CONDA_PREFIX} if length($ENV{CONDA_PREFIX} // '');
    push @candidates, '/usr';    # distribution packages
    my %seen;
    return grep { !$seen{$_}++ } @candidates;
}

sub find_cuda_home (@candidates) {
    for my $dir (@candidates) {
        return $dir if -f "$dir/include/cuda.h" && -f "$dir/include/nvrtc.h";
    }
    return;
}

# /usr/local/cuda-13.3 => 13.003, for sorting the newest first.
sub version_of ($dir) {
    my ($major, $minor) = $dir =~ /cuda-(\d+)(?:\.(\d+))?\z/ or return 0;
    return $major + ($minor // 0) / 1000;
}

sub which ($command) {
    for my $dir (File::Spec->path) {
        my $path = File::Spec->catfile($dir, $command);
        return $path if -f $path && -x _;
    }
    return;
}

1;
__END__

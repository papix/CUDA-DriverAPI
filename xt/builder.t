use v5.40;
use Test2::V0;
use File::Path qw(make_path);
use File::Temp qw(tempdir);
use FindBin;

# The toolkit lookup in builder/MyBuilder.pm, without building anything.
require "$FindBin::Bin/../builder/MyBuilder.pm";

my $root = tempdir(CLEANUP => 1);

sub toolkit ($dir, @headers) {
    make_path("$dir/include", "$dir/bin");
    for my $header (@headers) {
        open my $fh, '>', "$dir/include/$header" or die "$dir/include/$header: $!";
        close $fh;
    }
    return $dir;
}

subtest 'find_cuda_home needs both headers' => sub {
    my $partial = toolkit("$root/partial", 'cuda.h');
    my $full    = toolkit("$root/full",    'cuda.h', 'nvrtc.h');

    is builder::MyBuilder::find_cuda_home($partial, $full),  $full;
    is scalar(builder::MyBuilder::find_cuda_home($partial)), undef;
};

subtest 'an explicit CUDA_HOME or CUDA_PATH is the only candidate' => sub {
    local $ENV{CUDA_HOME} = "$root/explicit";
    is [ builder::MyBuilder::cuda_home_candidates() ], ["$root/explicit"];

    local $ENV{CUDA_HOME} = '';
    local $ENV{CUDA_PATH} = "$root/path";
    is [ builder::MyBuilder::cuda_home_candidates() ], ["$root/path"];
};

subtest 'without configuration, nvcc on PATH comes first and /usr last' => sub {
    my $toolkit = toolkit("$root/from-path", 'cuda.h', 'nvrtc.h');
    open my $fh, '>', "$toolkit/bin/nvcc" or die $!;
    close $fh;
    chmod 0755, "$toolkit/bin/nvcc";

    local $ENV{CUDA_HOME}    = '';
    local $ENV{CUDA_PATH}    = '';
    local $ENV{CONDA_PREFIX} = '';
    local $ENV{PATH}         = "$toolkit/bin";
    my @candidates = builder::MyBuilder::cuda_home_candidates();

    is $candidates[0],  $toolkit,          'the toolkit of nvcc on PATH';
    is $candidates[1],  '/usr/local/cuda', 'then the default install location';
    is $candidates[-1], '/usr',            'distribution packages last';
    is builder::MyBuilder::find_cuda_home(@candidates), $toolkit;
};

subtest 'nvcc reached through a symlink' => sub {
    my $toolkit = toolkit("$root/linked", 'cuda.h', 'nvrtc.h');
    open my $fh, '>', "$toolkit/bin/nvcc" or die $!;
    close $fh;
    chmod 0755, "$toolkit/bin/nvcc";
    make_path("$root/local/bin");
    symlink "$toolkit/bin/nvcc", "$root/local/bin/nvcc" or die $!;

    local $ENV{CUDA_HOME} = '';
    local $ENV{CUDA_PATH} = '';
    local $ENV{PATH}      = "$root/local/bin";
    my @candidates = builder::MyBuilder::cuda_home_candidates();
    is $candidates[0], Cwd::realpath($toolkit), 'where nvcc points first';
    ok + (grep { $_ eq "$root/local" } @candidates), 'then where it was found';
    is builder::MyBuilder::find_cuda_home(@candidates), Cwd::realpath($toolkit), 'the toolkit, not /local';
};

subtest 'Debian-style nvcc: the headers are next to the link, not the target' => sub {
    my $prefix = toolkit("$root/prefix", 'cuda.h', 'nvrtc.h');    # like /usr with /usr/include
    make_path("$root/prefix/lib/nvidia-cuda-toolkit/bin");
    open my $fh, '>', "$root/prefix/lib/nvidia-cuda-toolkit/bin/nvcc" or die $!;
    close $fh;
    chmod 0755, "$root/prefix/lib/nvidia-cuda-toolkit/bin/nvcc";
    symlink "$root/prefix/lib/nvidia-cuda-toolkit/bin/nvcc", "$prefix/bin/nvcc" or die $!;

    local $ENV{CUDA_HOME} = '';
    local $ENV{CUDA_PATH} = '';
    local $ENV{PATH}      = "$prefix/bin";
    is builder::MyBuilder::find_cuda_home(builder::MyBuilder::cuda_home_candidates()), $prefix;
};

subtest 'alternatives-style nvcc wins over stale headers next to the link' => sub {
    my $stale = toolkit("$root/stale", 'cuda.h', 'nvrtc.h');    # like /usr with old headers
    my $toolkit = toolkit("$root/cuda-13.3", 'cuda.h', 'nvrtc.h');
    open my $fh, '>', "$toolkit/bin/nvcc" or die $!;
    close $fh;
    chmod 0755, "$toolkit/bin/nvcc";
    symlink "$toolkit/bin/nvcc", "$stale/bin/nvcc" or die $!;

    local $ENV{CUDA_HOME} = '';
    local $ENV{CUDA_PATH} = '';
    local $ENV{PATH}      = "$stale/bin";
    is builder::MyBuilder::find_cuda_home(builder::MyBuilder::cuda_home_candidates()), Cwd::realpath($toolkit);
};

subtest 'versioned install directories, newest first' => sub {
    is [ sort { builder::MyBuilder::version_of($b) <=> builder::MyBuilder::version_of($a) }
            qw(/usr/local/cuda-12.9 /usr/local/cuda-13.10 /usr/local/cuda-13.3 /usr/local/cuda-12) ],
        [qw(/usr/local/cuda-13.10 /usr/local/cuda-13.3 /usr/local/cuda-12.9 /usr/local/cuda-12)];
};

done_testing;

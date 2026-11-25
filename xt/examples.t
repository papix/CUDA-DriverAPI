use v5.40;
use Test2::V0;
use File::Temp qw(tempdir);
use FindBin;
use IPC::Open3 qw(open3);

use CUDA::DriverAPI;

skip_all "CUDA device is not available: $@" unless eval { CUDA::DriverAPI->new };

# Run the examples in eg/ with small inputs so that the documentation stays
# working code.
my $eg    = "$FindBin::Bin/../eg";
my $cache = tempdir(CLEANUP => 1);

# No shell is involved, so paths with spaces or metacharacters are fine.
sub run_example (@args) {
    my $pid = open3(my $stdin, my $stdout, undef, $^X, '-Mblib', @args);
    local $/;
    my $output = <$stdout> // '';
    waitpid $pid, 0;
    is $?, 0, "@args exits successfully" or diag $output;
    return $output;
}

like run_example("$eg/axpy.pl", 1000), qr/y\[0\] = 3, y\[-1\] = 2001/;
like run_example("$eg/matmul.pl", 64, $cache), qr/kernel from NVRTC.*\nok/s;
like run_example("$eg/matmul.pl", 64, $cache), qr/kernel from cache.*\nok/s;

# A corrupt cache is discarded and rebuilt.
for my $file (glob "$cache/*.cubin") {
    open my $fh, '>:raw', $file or die "$file: $!";
    print {$fh} "\x7fELF truncated";
    close $fh;
}
like run_example("$eg/matmul.pl", 64, $cache), qr/Discarding the cached kernel.*kernel from NVRTC.*\nok/s;
like run_example("$eg/matmul.pl", 64, $cache), qr/kernel from cache.*\nok/s;

done_testing;

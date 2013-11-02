use strict;
use Test::More;

use CUDA::DeviceAPI;

is(CUDA::DeviceAPI::hello(), 'Hello, world!');

done_testing;


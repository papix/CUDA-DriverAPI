# Contributing to CUDA::DriverAPI

## Setup

- Perl 5.40 or later (CI covers 5.40, 5.42 and 5.44).
- The CUDA Toolkit 12.0 or later with NVRTC. `builder/MyBuilder.pm` finds it
  in the usual places (see INSTALLATION in the POD; `xt/builder.t` tests the
  lookup); set `CUDA_HOME` only to pick a specific one. On WSL, install only
  the toolkit; `libcuda` comes from the Windows driver through
  `/usr/lib/wsl/lib`.
- Dependencies, including the author tools:

  ```sh
  cpanm --with-configure --with-develop --installdeps .
  ```

## Build and test

```sh
perl Build.PL && ./Build && ./Build test   # unit tests; GPU tests skip without a device
minil test --release                       # plus POD, spelling, Perl::Critic, examples
perl -Mblib author/bench-array.pl          # array transfer benchmark
```

Tests that need a GPU call `skip_all` when `CUDA::DriverAPI->new` fails, so CI
(which has no GPU) only builds the module and runs the rest. Run the full
suite on a machine with an NVIDIA GPU before sending a change.

## Lint, formatting and secrets

```sh
author/lint.sh         # perltidy, perlcritic, actionlint, gitleaks (tree and history)
author/lint.sh --fix   # reformat Perl files with .perltidyrc first
```

It needs `perltidy` and `perlcritic` (develop dependencies), plus
[gitleaks](https://github.com/gitleaks/gitleaks) and
[actionlint](https://github.com/rhysd/actionlint). CI runs the same script.

`.pre-commit-config.yaml` checks staged changes for secrets and formatting on
every commit; enable it with `prek install` (or `pre-commit install`). Never
commit tokens, passwords, keys or `.env` files; `.gitignore` excludes the
usual names, and gitleaks catches the rest. A false positive goes into
`.gitleaksignore`, not around the hook.

C and XS are not auto-formatted (clang-format cannot parse XS); follow the
surrounding style and `.editorconfig`.

## Layout

| Path | Contents |
| --- | --- |
| `lib/CUDA/DriverAPI.xs` | Driver API and NVRTC calls, array conversion |
| `lib/CUDA/DriverAPI.pm` | Context object, `run`, `compile`, `load`, raw pointer API |
| `lib/CUDA/DriverAPI/Buffer.pm`, `Program.pm` | Objects returned by `buffer` and `compile`/`load` |
| `builder/MyBuilder.pm` | Finds the CUDA Toolkit and links `libcuda`/`libnvrtc` |
| `t/` | Tests; `t/ptx/*.ptx` are built from the `.cu` next to them |
| `xt/` | Author tests (Perl::Critic, examples) |
| `eg/`, `author/` | Examples and benchmarks |

## Generated files

Do not edit these by hand:

- `Build.PL`, `META.json` and `README.md` come from Minilla. Change `cpanfile`,
  `minil.toml` or the POD in `lib/CUDA/DriverAPI.pm`, then run `minil build`.
- `lib/CUDA/ppport.h` comes from `Devel::PPPort`.
- `t/ptx/*.ptx`: regenerate with
  `nvcc -ptx -arch=compute_75 t/ptx/NAME.cu -o t/ptx/NAME.ptx`.

## Style

- `use v5.40` with signatures; no experimental features.
- Every public method is documented in POD and has tests, including error
  cases.
- XS functions are private (`_name`) and wrapped by Perl methods that validate
  arguments; errors from the driver go through `CHECK_CU`/`CHECK_NVRTC`.
- Record user-visible changes in `Changes`.

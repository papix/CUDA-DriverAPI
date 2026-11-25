# AGENTS.md

Setup, commands, layout and style are in `CONTRIBUTING.md`; read it first.

## Checks

- `perl Build.PL && ./Build && ./Build test` after every change; `author/lint.sh`
  (run `author/lint.sh --fix` to apply perltidy) and `minil test --release`
  before committing.
- GPU tests skip silently without a device. A pass on a machine without a GPU,
  including CI, only proves the module builds.

## Decisions and pitfalls

- English only (the maintainer's decision): code, docs, commit messages, pull
  requests and issues. `author/lint.sh` flags Japanese text in files.
- No experimental Perl features (the maintainer's decision); stable `use v5.40`
  features only.
- `Build.PL`, `META.json`, `README.md` are generated: edit `cpanfile`,
  `minil.toml` or the POD, then `minil build`. Minilla does not copy dotfiles
  into its build directory, so author-test config lives in `xt/` (e.g.
  `xt/perlcriticrc`).
- Everything runs on the legacy default stream: `run` and `fill` are async and
  ordered before the synchronous copies. The XS resolves driver functions with
  `CU_GET_PROC_ADDRESS_LEGACY_STREAM` for this. Do not define
  `CUDA_API_PER_THREAD_DEFAULT_STREAM` or launch on other streams without
  adding explicit synchronization.
- All objects for a device share its primary context, so a sticky kernel error
  breaks every object in the process.
- libcuda is never linked: call driver functions as `CU(cuName)(...)` and add
  any new one to `CU_FUNCTIONS` in the XS, which resolves it with
  `cuGetProcAddress` when `new` first runs. A function newer than CUDA 12.0
  (like `cuFuncGetParamInfo`, 12.4) must stay optional, with a hand-written
  typedef, so that the minimum toolkit and driver stay at 12.0.
- Objects belong to the process that created them (`pid`): in a forked child,
  methods croak and releasing calls no driver function.
- The driver loads code lazily: `load` of a broken image succeeds and fails at
  the first `run` (`LibraryGetKernel: CUDA_ERROR_INVALID_IMAGE`).
- perltidy skips the generated `Build.PL`; never reformat it.
- PDL is optional (the maintainer's decision): the same `write_array`/`read_array`
  calls take nested arrays or ndarrays, nothing loads PDL unless an ndarray or
  `as => 'pdl'` is involved (`t/10_without_pdl.t`), and only PDL's Perl API
  (`get_dataref`/`upd_data`) is used, never its C headers.
- Keep existing method semantics; the downstream project papix/PerCUDA uses
  `buffer`, `write_array`, `read_array`, `fill`, `load`, `compute_capability`
  and `run` with `[type => value]` arguments.

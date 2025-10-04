WIP: Early Windows packaging documentation

Prerequisite is a environment set up as described in [Windows build instructions](../../docs/windows_install.md).

```powershell
 pyinstaller lada.spec
```

On my Windows build machine the pyinstaller command crashes caused by polars dependency pulled in by ultralytics.
Problem seems to be that it expects AVX512 capable CPU which this machine doesn't offer. Fortunately there is an alternative package:

```powershell
pip uninstall polars
pip install polars-lts-cpu
```


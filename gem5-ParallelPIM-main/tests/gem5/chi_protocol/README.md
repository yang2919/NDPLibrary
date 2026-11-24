# CHI Protocol with different ISAs test

The purpose of this test is to ensure the CHI protocol functions with the RISCV ISA target with varying numbers of CPU cores. The tests can be run with the following command:

```shell
# In the "tests" directory
./main.py run --length=long -j`nproc` gem5/chi_protocol
```

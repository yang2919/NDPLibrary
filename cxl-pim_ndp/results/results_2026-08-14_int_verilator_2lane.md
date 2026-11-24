# Integer ISA Tests, Verilator (2026-08-14)

| | |
|---|---|
| Simulator | Verilator, 2 lanes, VLEN 2048 |
| RTL | `rtl/ara` as of 2026-08-14 |

| Group | Test | Result |
|---|---|---|
| mul | `vmul` | PASS |
| mul | `vmulh` | PASS |
| mul | `vmulhu` | PASS |
| mul | `vmulhsu` | PASS |
| mul | `vwmul` | PASS |
| mul | `vwmulu` | PASS |
| mul | `vwmulsu` | PASS |
| mul | `vmacc` | PASS |
| mul | `vnmsac` | PASS |
| mul | `vmadd` | PASS |
| mul | `vnmsub` | PASS |
| mul | `vwmacc` | PASS |
| mul | `vwmaccu` | PASS |
| mul | `vwmaccsu` | PASS |
| mul | `vwmaccus` | PASS |
| div | `vdiv` | PASS |
| div | `vdivu` | PASS |
| div | `vrem` | PASS |
| div | `vremu` | PASS |
| fixpt | `vsadd` | PASS |
| fixpt | `vsaddu` | PASS |
| fixpt | `vaadd` | PASS |
| fixpt | `vsmul` | PASS |
| fixpt | `vssrl` | PASS |
| fixpt | `vssra` | PASS |
| fixpt | `vnclip` | PASS |
| fixpt | `vnclipu` | PASS |
| alu | `vadd` | PASS |
| alu | `vsub` | PASS |
| alu | `vrsub` | PASS |
| alu | `vand` | PASS |
| alu | `vor` | PASS |
| alu | `vxor` | PASS |
| alu | `vsll` | PASS |
| alu | `vsrl` | PASS |
| alu | `vsra` | PASS |
| alu | `vmax` | PASS |
| alu | `vmaxu` | PASS |
| alu | `vmin` | PASS |
| alu | `vminu` | PASS |
| alu | `vmseq` | PASS |
| alu | `vmsne` | PASS |
| alu | `vmslt` | PASS |
| alu | `vmsltu` | PASS |
| alu | `vmsle` | PASS |
| alu | `vmsleu` | PASS |
| alu | `vmsgt` | PASS |
| alu | `vmsgtu` | PASS |
| alu | `vadc` | PASS |
| alu | `vsbc` | PASS |
| alu | `vmadc` | PASS |
| alu | `vmsbc` | PASS |
| alu | `vmerge` | PASS |
| red | `vredsum` | PASS |
| red | `vredmax` | PASS |
| red | `vredmaxu` | PASS |
| red | `vredmin` | PASS |
| red | `vredminu` | PASS |
| red | `vredand` | PASS |
| red | `vredor` | PASS |
| red | `vredxor` | PASS |
| red | `vwredsum` | PASS |
| red | `vwredsumu` | PASS |
| conv | `vsext` | PASS |
| conv | `vzext` | PASS |
| conv | `vnsra` | PASS |
| conv | `vnsrl` | PASS |

**67 PASS, 0 FAIL, 0 missing.**

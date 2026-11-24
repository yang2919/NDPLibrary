# Vivado Reports

Every report comes from the implementation run that produced the bitstream in [`../pdi/`](../pdi/).

| File | Vivado report |
|---|---|
| `timing_summary.rpt` | Post-route physical optimization timing summary |
| `utilization.rpt` | Utilization, device totals and by hierarchy |
| `route_status.rpt` | Routing completion and error count |
| `clock_utilization.rpt` | Clock resource usage |
| `power.rpt` | Power estimate |
| `methodology.rpt` | Methodology DRC |

Summary of `timing_summary.rpt` and `utilization.rpt`:

| | |
|---|---|
| WNS / TNS | 0.000 ns / 0.000 ns, 0 failing of 239,045 endpoints |
| WHS / THS | 0.008 ns / 0.000 ns, 0 failing of 239,045 endpoints |
| WPWS / TPWS | 0.000 ns / 0.000 ns, 0 failing of 81,954 endpoints |
| Nets with routing errors | 0 of 309,880 |
| CLB LUTs | 219,423 of 1,753,984 (12.51 %) |
| CLB Registers | 81,370 of 3,507,968 (2.32 %) |
| Block RAM tile | 60 of 2,541 (2.36 %) |
| URAM288 | 32 of 1,301 (2.46 %) |
| DSP58 | 104 of 7,392 (1.41 %) |

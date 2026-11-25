# Stock gem5 FS verification

Use the stock x86 Linux guest to reproduce the software receiver checks. F2 covers metadata delivery, F3 persistent emulator execution, F4 Session lifetime and fault handling, and F5 the decoder. For RISC-V PIM instruction execution use the [extended simulator procedure](extended-gem5-smoke-verification.md).

## Run a stage

Start Docker and run one stage at a time from the repository root. The runner reuses the image and build resources. See the [F0 baseline](gem5-fs.md) for prerequisites and boot diagnostics.

```sh
stage=F2  # choose F2, F3, F4 or F5
before=$(cat build-gem5-fs/latest-run.txt 2>/dev/null || true)
STAGE="$stage" BOOT_TIMEOUT=3600 ./tools/gem5/fs/run.sh
status=$?
after=$(cat build-gem5-fs/latest-run.txt)
run_dir="build-gem5-fs/$(basename "$after")"
printf '%s\n' "$status" "$before" "$after"
```

Require exit status zero and a fresh directory named `run-<stage>-…`. In that directory, check `full_system=true` in `config.ini`, the stage verdict in `result.txt` and the guest markers in `board.pc.com_1.device`. If the run fails, inspect its `console.log` and guest serial log. If setup failed before it created a directory, use the terminal output.

## Expected verdict and serial markers

### F2

[Stage implementation](gem5-f2.md)

```text
F2 PASS: Guest metadata IPC verified; no device execution.
NDP_METADATA_ACCEPTED batches=3 exact_bytes=yes
NDP_METADATA_REJECTIONS duplicate_order_epoch_opcode_range=5
NDP_METADATA_ONLY execution=not_performed
NDP_F2_PASS
NDP_FS_SCRIPT_COMPLETE
```

### F3

[Stage implementation](gem5-f3.md)

```text
F3 PASS: Persistent guest emulator execution verified.
NDP_EXECUTION_COMPLETE pnm=pim=verified
NDP_PERSISTENT_STATE mac_memory=verified
NDP_REFERENCE_MATCH exact_fp16=yes
NDP_EXECUTION_ERRORS preflight_download_recovery=verified
NDP_F3_PASS
NDP_FS_SCRIPT_COMPLETE
```

### F4

[Stage implementation](gem5-f4.md)

```text
F4 PASS: Session lifecycle and receiver fault cleanup verified.
NDP_SESSION_DEPENDENCIES pnm_pim=verified
NDP_RESIDENT_BUFFERS copy_fill_snapshot=verified
NDP_SESSION_CLOSE drain_live_handles_reaped=verified
NDP_SESSION_FAULTS timeout_disconnect_protocol=verified
NDP_SESSION_LIFETIME idle_destructor=verified
NDP_F4_PASS
NDP_FS_SCRIPT_COMPLETE
```

### F5

[Stage implementation](gem5-f5.md)

```text
F5 PASS: Guest decoder, KV reset and memory reclamation verified.
NDP_DECODER_EXECUTION prefill_pnm=3 decode_pim=9
NDP_DECODER_REFERENCE outputs=12 exact_fp16=yes
NDP_DECODER_REQUESTS sequence=A_B_A kv_reset=verified
NDP_DECODER_RECLAMATION temporaries_model_cache_workspace=verified
NDP_F5_PASS
NDP_FS_SCRIPT_COMPLETE
```

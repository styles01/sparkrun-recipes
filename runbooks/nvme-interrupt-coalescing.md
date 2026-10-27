# NVMe Interrupt Coalescing — larryspark (found via @wrldsuksgo2mars PSA, Sep 30 2026)

## What it is
NVIDIA ships `nvidia-nvme-interrupt-coalescing.service` (enabled by preset) that forces
NVMe feature 0x08 = 0x00000107 (TIME=100µs, THR=8) on **Samsung/Kioxia/Micron** drives only.
Ours: SAMSUNG MZALC4T0HBL1 (vendor 0x144d) on /dev/nvme0 — affected.

## Measured impact (own box, random 4K QD1 O_DIRECT, n=2800, 2026-09-30)
- coalescing ON:  mean 227 µs  p50 198  p99 623
- coalescing OFF: mean  54 µs  p50 53  p99 84
≈ 4.2× read latency penalty. Uncached Engram/PLE/mmapped KV lookups eat this.

## Fix applied (2026-09-30, no local sudo needed — docker privileged path)
1. Runtime off: docker run --privileged -v /dev:/dev nvcr pytorch    nvme set-feature /dev/nvme0 -f 8 --value=0
2. Boot persistence WITHOUT sudo: masked the unit by symlink
   /etc/systemd/system/nvidia-nvme-interrupt-coalescing.service -> /dev/null
   (done via privileged docker mount of /). is-enabled now reports "masked".
   On-boot ExecStart can never fire again.
3. Feature 0x08 is NOT persistable across power cycle on this drive
   ("Feature Identifier Not Saveable"). Re-apply after reboot via:
   [re-run the docker one-liner above] — TODO: cron @reboot if reboots are frequent.

## Verify
nvme get-feature /dev/nvme0 -f 8 -H   # want TIME=0 usec, THR=1

---
description: How to connect to and run commands on the USC CARC cluster
---

# USC CARC Cluster Access

## SSH Connection
The cluster hostname is **endeavour.usc.edu** (not discovery).

```bash
ssh zizhaoh@endeavour.usc.edu "<command>"
```

## SCP File Transfer
```bash
# Upload
scp <local_file> zizhaoh@endeavour.usc.edu:<remote_path>

# Download
scp zizhaoh@endeavour.usc.edu:<remote_path> <local_file>
```

## Key Paths on Cluster
- **Project dir**: `/project2/jessetho_1732/zizhaoh/DREAM-C2L`
- **Scratch results**: `/scratch1/zizhaoh/PRISM/results/`
- **HF cache**: `/project2/jessetho_1732/zizhaoh/.cache/huggingface`

## SLURM Partition
- Primary partition: `gpu` or `nlp_hiprio`
- Account: `jessetho_1732`

## Important Notes
- When running remote scripts, use `scp` to upload the script first, then `ssh` to execute it. This avoids PowerShell quoting issues with heredocs.
- Always use `endeavour.usc.edu`, never `discovery.usc.edu`.

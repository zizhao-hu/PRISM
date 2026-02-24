"""Extract clean log lines from SLURM stderr (strips \\r progress bars)."""
import sys
with open(sys.argv[1], 'rb') as f:
    data = f.read()

# Replace \r\n with \n, then split each \r-delimited chunk and take the last part
data = data.replace(b'\r\n', b'\n')
lines = data.split(b'\n')
clean = []
for line in lines:
    # For lines with \r (tqdm progress), take the last segment
    parts = line.split(b'\r')
    last = parts[-1].decode('utf-8', 'replace').strip()
    if last:
        clean.append(last)

# Print last 50 clean lines
for line in clean[-50:]:
    print(line)

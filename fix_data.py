import re
from collections import defaultdict
from pathlib import Path

def fix_data_file(input_path, output_path=None):
    # Regex to match lines like: 00:11:10.031=yes
    entry_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})=(\w+)")
    
    # Collect all data: { filename: { timestamp: label } }
    data = defaultdict(dict)
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            filename, entries = map(str.strip, line.split("|", 1))
            for match in entry_pattern.finditer(entries):
                timestamp, label = match.groups()
                data[filename][timestamp] = label  # overwrite duplicates

    # Sort timestamps and rebuild clean lines
    fixed_lines = []
    for filename in sorted(data.keys()):
        timestamps = sorted(data[filename].keys())
        entries_str = ", ".join(f"{ts}={data[filename][ts]}" for ts in timestamps)
        fixed_lines.append(f"{filename} | {entries_str}")

    # Write out cleaned file
    if not output_path:
        output_path = Path(input_path).with_name("fixed_" + Path(input_path).name)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in fixed_lines:
            f.write(line + "\n")

    print(f"✅ Fixed data written to {output_path}")


# Example usage:
fix_data_file(r"D:\NewFolder(3)\videoProject\_frame_annotations.txt")

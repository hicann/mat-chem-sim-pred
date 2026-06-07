import sys
import struct
import os

# Compute build directory relative to this script (project root build/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

# Find .ascend.kernel section in the .so (skip 20-byte CANN header)
so_path = os.path.join(_PROJECT_ROOT, "build", "lib", "libgaff2_force_kernel.so")
with open(so_path, "rb") as f:
    full = f.read()

idx = full.find(b'\x7fELF')
sub = full[idx:]
e_shoff = struct.unpack('<Q', sub[40:48])[0]
e_shentsize = struct.unpack('<H', sub[58:60])[0]
e_shnum = struct.unpack('<H', sub[60:62])[0]
e_shstrndx = struct.unpack('<H', sub[62:64])[0]

str_tab_hdr = e_shoff + e_shstrndx * e_shentsize
str_offset = struct.unpack('<Q', sub[str_tab_hdr+24:str_tab_hdr+32])[0]
str_size = struct.unpack('<Q', sub[str_tab_hdr+32:str_tab_hdr+40])[0]
str_table = sub[str_offset:str_offset+str_size]

data = None
for s in range(e_shnum):
    sh_off = e_shoff + s * e_shentsize
    name_idx = struct.unpack('<I', sub[sh_off:sh_off+4])[0]
    sh_offset = struct.unpack('<Q', sub[sh_off+24:sh_off+32])[0]
    sh_size = struct.unpack('<Q', sub[sh_off+32:sh_off+40])[0]
    name = str_table[name_idx:str_table.find(b'\x00', name_idx)].decode('ascii', errors='replace')
    if 'ascend.kernel' in name:
        data = sub[sh_offset:sh_offset+sh_size]
        break

if data is None:
    print("ERROR: Could not find .ascend.kernel section in .so", file=sys.stderr)
    sys.exit(1)

# The .so section has a 20-byte header (CANN metadata), skip it
# The actual ELF binary starts at offset 20
if len(data) > 20 and data[0] != 0x7f and data[20] == 0x7f:
    data = data[20:]

print(f"Extracted {len(data)} bytes from {so_path}")

out_path = os.path.join(_SCRIPT_DIR, "gaff2_device_aiv_bin.inc")
with open(out_path, "w") as out:
    out.write("// Auto-generated from {}\n".format(so_path))
    out.write("// {} bytes (skipped 20-byte CANN header)\n".format(len(data)))
    for i in range(0, len(data)):
        if i % 12 == 0:
            out.write("  ")
        out.write("0x{:02x}".format(data[i]))
        if i < len(data) - 1:
            out.write(", ")
        else:
            out.write("\n")
        if i % 12 == 11:
            out.write("\n")

print("Written inc file: {} bytes".format(len(data)))

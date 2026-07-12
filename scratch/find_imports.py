import struct
import sys

def get_imported_dlls(pe_path):
    try:
        with open(pe_path, "rb") as f:
            data = f.read()
            
        # Parse DOS header
        if data[:2] != b"MZ":
            return ["Not a valid PE file"]
            
        pe_offset = struct.unpack_value_at(data, 0x3c)
        if data[pe_offset:pe_offset+4] != b"PE\x00\x00":
            return ["Invalid PE signature"]
            
        # Parse PE header (COFF)
        magic = struct.unpack_value_at(data, pe_offset + 24)
        is_pe32_plus = (magic == 0x20b)
        
        # Directory offsets in optional header
        dir_offset = pe_offset + (120 if is_pe32_plus else 104)
        import_table_rva, import_table_size = struct.unpack("<II", data[dir_offset:dir_offset+8])
        
        if import_table_size == 0:
            return []
            
        # Find section containing the Import Table RVA
        num_sections = struct.unpack("<H", data[pe_offset+6:pe_offset+8])[0]
        optional_header_size = struct.unpack("<H", data[pe_offset+20:pe_offset+22])[0]
        section_table_offset = pe_offset + 24 + optional_header_size
        
        import_section = None
        for i in range(num_sections):
            offset = section_table_offset + i * 40
            sec_name = data[offset:offset+8].rstrip(b"\x00")
            vsize = struct.unpack("<I", data[offset+8:offset+12])[0]
            rva = struct.unpack("<I", data[offset+12:offset+16])[0]
            raw_size = struct.unpack("<I", data[offset+16:offset+20])[0]
            raw_ptr = struct.unpack("<I", data[offset+20:offset+24])[0]
            
            if rva <= import_table_rva < rva + vsize:
                import_section = (rva, raw_ptr)
                break
                
        if not import_section:
            return ["Import table section not found"]
            
        rva_start, raw_ptr_start = import_section
        import_table_offset = raw_ptr_start + (import_table_rva - rva_start)
        
        # Read import directory descriptors
        dlls = []
        offset = import_table_offset
        while True:
            desc = data[offset:offset+20]
            if desc == b"\x00" * 20:
                break
            name_rva = struct.unpack("<I", desc[12:16])[0]
            name_offset = raw_ptr_start + (name_rva - rva_start)
            
            # Read null-terminated string
            dll_name = []
            while data[name_offset] != 0:
                dll_name.append(chr(data[name_offset]))
                name_offset += 1
            dlls.append("".join(dll_name))
            offset += 20
            
        return dlls
    except Exception as e:
        return [f"Error parsing PE: {e}"]

# Helper to unpack integer from bytes
def unpack_value_at(data, offset, fmt="<I"):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, data[offset:offset+size])[0]

# Attach helper
struct.unpack_value_at = unpack_value_at

if __name__ == "__main__":
    import os
    pe_path = os.path.join(
        "C:\\Users\\qcwor\\Documents\\qualcomm-hack-lore\\.venv\\Lib\\site-packages\\onnxruntime_qnn",
        "QnnHtpV73Stub.dll"
    )
    print("DLL Imports:", get_imported_dlls(pe_path))

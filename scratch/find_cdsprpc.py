import os
import glob

def find_dll():
    # Search for libcdsprpc.dll in Windows DriverStore
    pattern = 'C:/Windows/System32/DriverStore/FileRepository/**/libcdsprpc.dll'
    print("Searching for libcdsprpc.dll in DriverStore...")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        print("Found matches:")
        for m in matches:
            print(f"- {m}")
    else:
        # Search the whole C:/Windows directory for it
        print("Not found in DriverStore. Searching C:/Windows/...")
        pattern_win = 'C:/Windows/**/libcdsprpc.dll'
        matches_win = glob.glob(pattern_win, recursive=True)
        if matches_win:
            print("Found matches in C:/Windows:")
            for m in matches_win:
                print(f"- {m}")
        else:
            print("libcdsprpc.dll not found anywhere on the system.")

if __name__ == "__main__":
    find_dll()

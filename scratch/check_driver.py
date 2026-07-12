import subprocess

def check():
    cmd = [
        "powershell",
        "-Command",
        "Get-CimInstance Win32_PnPSignedDriver | Where-Object { $_.FriendlyName -like '*Compute DSP*' } | Select-Object FriendlyName, DriverVersion, Manufacturer | ConvertTo-Json"
    ]
    try:
        out = subprocess.check_output(cmd).decode()
        print(out)
    except Exception as e:
        print("Error executing query:", e)

if __name__ == "__main__":
    check()

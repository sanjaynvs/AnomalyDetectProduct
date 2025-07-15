import subprocess

def clear_time_wait_on_port(port=8000):
    print(f"🔍 Checking for TIME_WAIT connections on port {port}...")
    result = subprocess.run(
        f'netstat -aon | findstr :{port}',
        shell=True,
        capture_output=True,
        text=True
    )

    lines = result.stdout.splitlines()
    pids_to_kill = set()

    for line in lines:
        parts = line.split()
        if len(parts) == 5 and parts[3] == 'TIME_WAIT':
            pid = parts[4]
            if pid != '0':
                pids_to_kill.add(pid)

    if not pids_to_kill:
        print("✅ No active TIME_WAIT sockets held by user-space processes.")
        return

    print(f"⚠️ Killing PIDs holding TIME_WAIT: {', '.join(pids_to_kill)}")

    for pid in pids_to_kill:
        subprocess.run(f'taskkill /PID {pid} /F', shell=True)

    print("🧹 Cleanup complete!")

clear_time_wait_on_port()
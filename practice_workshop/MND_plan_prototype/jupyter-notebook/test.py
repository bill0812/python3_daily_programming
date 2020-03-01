import shlex, subprocess

result = subprocess.Popen(shlex.split("potrace --svg result.pnm -o 2.svg"), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
result.wait()

print(result.returncode)

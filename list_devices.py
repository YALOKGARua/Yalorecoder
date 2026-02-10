import sounddevice as sd

print("=== Audio Devices ===")
devices = sd.query_devices()
for i, d in enumerate(devices):
    print(f"  [{i}] {d['name']}")
    print(f"      in={d['max_input_channels']} out={d['max_output_channels']} api={d['hostapi']} sr={d['default_samplerate']}")

print()
print("=== Host APIs ===")
apis = sd.query_hostapis()
for i, a in enumerate(apis):
    print(f"  [{i}] {a['name']} (devices: {a['devices']})")

print()
try:
    default_in = sd.query_devices(kind='input')
    print(f"Default input: {default_in['name']}")
except:
    print("No default input device")

try:
    default_out = sd.query_devices(kind='output')
    print(f"Default output: {default_out['name']}")
except:
    print("No default output device")

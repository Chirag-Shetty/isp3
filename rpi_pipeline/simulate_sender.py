import sys
import json
import time

def simulate_frame_generator(json_path):
    """
    Reads a replay JSON file (like replay_1.json) and yields
    frames to simulate the radar hardware.
    """
    print(f"[simulator] Loading dataset {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data_file = json.load(f)
    
    # Check if 'data' array exists (which holds frames in replay_*.json)
    if 'data' not in data_file:
        print("[simulator] Invalid JSON format. Missing 'data' array.")
        return

    frames = data_file['data']
    print(f"[simulator] Found {len(frames)} frames. Starting continuous playback...")
    
    while True:
        for i, row in enumerate(frames):
            fd = row.get("frameData", {})
            
            # Build the same dict format that radar_capture now returns
            frame_dict = {
                "pointCloud": fd.get("pointCloud", []),
                "trackData": fd.get("trackData", []),
                "heightData": fd.get("heightData", [])
            }
            yield frame_dict
            
            # Simulate real-time delay (e.g. 55ms per frame)
            time.sleep(0.055)


def main():
    if len(sys.argv) < 2:
        print("Usage: python simulate_sender.py <path_to_replay_json>")
        print(r"Example: python simulate_sender.py ..\dataset_26_09_25 (1)\dataset_26_09_25\chair_floor_transition\replay_1.json")
        sys.exit(1)
        
    json_path = sys.argv[1]

    # Dynamically monkey-patch the config so it knows we're simulating
    import config
    config.RUN_INFERENCE = True # Ensure inference runs locally or remote
    
    # Import rpi_sender dynamically to swap out the hardware generator
    import rpi_sender
    
    # 1. Provide a dummy module config function so we don't try serial commands
    def dummy_config(port):
        print("[simulator] Skipped sending hardware config to port.")
    rpi_sender.send_config = dummy_config
    
    # 2. Swap out the frame_generator for our simulator
    def wrapper(port):
        yield from simulate_frame_generator(json_path)
    rpi_sender.frame_generator = wrapper
    
    # 3. Suppress URL checks if needed, but let's let it run
    print("[simulator] ===============================================")
    print(f"[simulator] Running RPi pipeline in SIMULATOR mode")
    print("[simulator] ===============================================")
    
    # Call the main loop!
    rpi_sender.main()

if __name__ == '__main__':
    main()

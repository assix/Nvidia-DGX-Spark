#!/usr/bin/env python3

import subprocess
import time
import sys
import re

def get_gpu_stats():
    """
    Queries nvidia-smi for power, utilization, and graphics clock.
    Returns a tuple: (power_w, util_percent, clock_mhz)
    Returns (None, None, None) if parsing fails.
    """
    try:
        # Query for power, utilization, and graphics clock.
        # Removed 'utilization.tensor' which was causing the error.
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=power.draw,utilization.gpu,clocks.current.graphics',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Clean the output (removes extra spaces) and split by comma
        cleaned_output = re.sub(r'\s+', '', result.stdout.strip())
        parts = cleaned_output.split(',')

        if len(parts) == 3:
            power = float(parts[0])
            util = float(parts[1])
            clock = float(parts[2])
            return power, util, clock
        else:
            print(f"Error: Unexpected nvidia-smi output: '{result.stdout}'", file=sys.stderr)
            return None, None, None

    except FileNotFoundError:
        print("Error: 'nvidia-smi' command not found.", file=sys.stderr)
        sys.exit(1)
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        # Print the exact error if subprocess fails
        print(f"Error parsing nvidia-smi stats: {e}", file=sys.stderr)
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return None, None, None

def main():
    print("Starting GPU performance monitoring...")
    print("Run your inference script now.")
    print("Press Ctrl+C to stop monitoring and see results.")
    print("-" * 60)
    
    # Print 3 empty lines to reserve the space for the live output
    print("\n\n") 

    power_readings = []
    util_readings = []
    clock_readings = []
    
    max_power_seen = 0.0
    max_util_seen = 0.0
    max_clock_seen = 0.0
    
    interval_seconds = 1.0

    try:
        while True:
            power, util, clock = get_gpu_stats()
            
            if power is not None and util is not None and clock is not None:
                # Update peak stats
                if power > max_power_seen: max_power_seen = power
                if util > max_util_seen: max_util_seen = util
                if clock > max_clock_seen: max_clock_seen = clock
                
                # Store readings for final report
                power_readings.append(power)
                util_readings.append(util)
                clock_readings.append(clock)
                
                # --- Multi-line live display ---
                # \033[3A = Move cursor UP 3 lines
                print("\033[3A", end="")
                
                # \033[K = Clear to end of line (prevents artifacts)
                print(f"Power: {power:6.2f} W (Peak: {max_power_seen:6.2f} W) \033[K")
                print(f"Util:  {util:6.0f} % (Peak: {max_util_seen:6.0f} %) \033[K")
                print(f"Clock: {clock:6.0f} MHz (Peak: {max_clock_seen:6.0f} MHz)\033[K")
                
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        # Move cursor down 3 lines to be below the live output
        print("\n\n\n" + "-" * 60)
        print("Monitoring stopped.")
        
        if not power_readings:
            print("No power readings were captured.")
            return

        # Calculate statistics
        print("\n--- Performance Report (Average) ---")
        if power_readings:
            print(f"  Power: Avg {sum(power_readings) / len(power_readings):.2f} W (Min: {min(power_readings):.2f}, Max: {max(power_readings):.2f})")
        if util_readings:
            print(f"  Util:  Avg {sum(util_readings) / len(util_readings):.2f} % (Min: {min(util_readings):.0f}, Max: {max(util_readings):.0f})")
        if clock_readings:
            print(f"  Clock: Avg {sum(clock_readings) / len(clock_readings):.2f} MHz (Min: {min(clock_readings):.0f}, Max: {max(clock_readings):.0f})")
        print("--------------------------------------")

if __name__ == "__main__":
    main()
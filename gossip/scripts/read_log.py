import json
import numpy as np
import argparse
from datetime import datetime

"""
Example JSON:
{"timestamp": "2025-10-25T23:30:01.123Z", "latency_ms": 110, "queue_depth": 5, "request_id": "a-123"}
{"timestamp": "2025-10-25T23:30:02.456Z", "latency_ms": 85, "queue_depth": 4, "request_id": "b-456"}
{"timestamp": "2025-10-25T23:30:02.789Z", "latency_ms": 250, "queue_depth": 6, "request_id": "c-789"}
{"timestamp": "2025-10-25T23:30:03.100Z", "latency_ms": 120, "queue_depth": 5, "request_id": "d-012"}
"""

def parse_iso_timestamp(ts_str):
    """Handles ISO 8601 timestamps, including 'Z' for UTC."""
    if ts_str.endswith('Z'):
        ts_str = ts_str[:-1] + '+00:00'
    return datetime.fromisoformat(ts_str)

def calculate_metrics(logfile_path):
    """
    Reads a log file and calculates P95/P99 latency, queue depth std dev,
    and throughput.
    """
    latencies = []
    queue_depths = []
    request_count = 0
    first_timestamp = None
    last_timestamp = None

    print(f"Processing log file: {logfile_path}...")

    try:
        with open(logfile_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    log_entry = json.loads(line)

                    # --- 1. Collect Latency ---
                    if 'latency_ms' in log_entry:
                        latencies.append(log_entry['latency_ms'])

                    # --- 2. Collect Queue Depth ---
                    if 'queue_depth' in log_entry:
                        queue_depths.append(log_entry['queue_depth'])
                    
                    # --- 3. Collect Data for Throughput ---
                    if 'timestamp' in log_entry:
                        request_count += 1
                        timestamp = parse_iso_timestamp(log_entry['timestamp'])
                        
                        if first_timestamp is None or timestamp < first_timestamp:
                            first_timestamp = timestamp
                        if last_timestamp is None or timestamp > last_timestamp:
                            last_timestamp = timestamp

                except json.JSONDecodeError:
                    print(f"Warning: Skipping non-JSON line: {line[:50]}...")
                except KeyError as e:
                    print(f"Warning: Skipping line with missing key {e}: {line[:50]}...")
                except Exception as e:
                    print(f"Warning: Skipping line due to error: {e}")

    except FileNotFoundError:
        print(f"Error: Log file not found at {logfile_path}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # --- Perform Calculations ---
    print("\n--- Performance Metrics ---")

    # Latency Metrics
    if latencies:
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        avg_latency = np.mean(latencies)
        print(f"Latency (ms):")
        print(f"  - Average: {avg_latency:.2f}")
        print(f"  - P95:     {p95_latency:.2f}")
        print(f"  - P99:     {p99_latency:.2f}")
    else:
        print("Latency (ms): No 'latency_ms' data found.")

    # Queue Depth Metrics
    if queue_depths:
        std_dev_queue = np.std(queue_depths)
        avg_queue = np.mean(queue_depths)
        print(f"\nQueue Depth:")
        print(f"  - Average:        {avg_queue:.2f}")
        print(f"  - Std. Deviation: {std_dev_queue:.2f}")
    else:
        print("\nQueue Depth: No 'queue_depth' data found.")

    # Throughput Metrics
    if request_count > 0 and first_timestamp and last_timestamp:
        total_seconds = (last_timestamp - first_timestamp).total_seconds()
        
        # Avoid division by zero if all logs are in the same second
        if total_seconds < 1.0:
            total_seconds = 1.0 
            
        throughput_rps = request_count / total_seconds
        print(f"\nThroughput:")
        print(f"  - Total Requests: {request_count}")
        print(f"  - Time Window (sec): {total_seconds:.2f}")
        print(f"  - Avg. Throughput (RPS): {throughput_rps:.2f}")
    else:
        print("\nThroughput: Not enough data to calculate.")


def main():
    parser = argparse.ArgumentParser(
        description="Process application logs for performance metrics."
    )
    parser.add_argument(
        "logfile", 
        help="Path to the raw log file (JSONL format)."
    )
    args = parser.parse_args()
    
    calculate_metrics(args.logfile)

if __name__ == "__main__":
    main()

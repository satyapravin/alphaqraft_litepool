#!/usr/bin/env python3
"""
Resample CSV files to 100 ms intervals.
For each 100 ms interval, uses the next available timestamp's data (no interpolation).
Uses streaming approach to avoid loading entire file into memory.
"""

import csv
import os
import sys
from pathlib import Path

def parse_timestamp(ts_str):
    """Parse timestamp string to microseconds."""
    return int(ts_str)

def resample_csv(input_file, output_file, interval_ms=100):
    """
    Resample CSV to fixed intervals using streaming (memory-efficient).
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        interval_ms: Interval in milliseconds (default 100 ms)
    """
    interval_us = interval_ms * 1000  # Convert to microseconds
    
    print(f"Resampling {input_file} to {interval_ms} ms intervals...")
    
    # First pass: get first timestamp and fieldnames
    first_ts = None
    fieldnames = None
    
    with open(input_file, 'r') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        
        # Read first valid row to get first timestamp
        for row in reader:
            try:
                first_ts = parse_timestamp(row['timestamp'])
                break
            except (KeyError, ValueError):
                continue
    
    if first_ts is None:
        print(f"Error: No valid rows in {input_file}")
        return
    
    if fieldnames is None:
        print(f"Error: Could not read fieldnames from {input_file}")
        return
    
    print(f"  First timestamp: {first_ts}")
    
    # Second pass: stream through file and resample
    output_count = 0
    current_target_ts = first_ts
    last_row = None
    last_row_ts = None
    
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            try:
                row_ts = parse_timestamp(row['timestamp'])
            except (KeyError, ValueError):
                continue
            
            # Keep track of the last row we've seen
            last_row = row
            last_row_ts = row_ts
            
            # While current row timestamp >= current target, write it and advance target
            while row_ts >= current_target_ts:
                # Write this row with updated timestamp
                resampled_row = row.copy()
                resampled_row['timestamp'] = str(current_target_ts)
                writer.writerow(resampled_row)
                output_count += 1
                
                # Move to next target timestamp
                current_target_ts += interval_us
                
                # If we've passed the last row, we're done
                if row_ts < current_target_ts:
                    break
        
        # If we have a last row and haven't reached it yet, use it for remaining targets
        if last_row is not None and last_row_ts is not None:
            while current_target_ts <= last_row_ts:
                resampled_row = last_row.copy()
                resampled_row['timestamp'] = str(current_target_ts)
                writer.writerow(resampled_row)
                output_count += 1
                current_target_ts += interval_us
    
    print(f"  Output: {output_count} rows")
    print(f"  Written to {output_file}")

def main():
    """Main function to resample all CSV files in data/training/"""
    data_dir = Path("/home/pravin/dev/alphaqraft_litepool/data/training")
    output_dir = data_dir / "resampled_100ms"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = sorted(data_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to resample\n")
    
    for csv_file in csv_files:
        output_file = output_dir / csv_file.name
        resample_csv(csv_file, output_file, interval_ms=100)
        print()
    
    print(f"Resampling complete! Output files in: {output_dir}")

if __name__ == "__main__":
    main()



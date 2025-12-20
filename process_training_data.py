#!/usr/bin/env python3
"""
Process training data: unzip, resample to 100ms, and rename files.

For books: resample to 100ms intervals (similar to resample_csv.py)
For trades: aggregate to 100ms intervals, determine side based on price movement
"""

import csv
import gzip
import os
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def parse_timestamp(ts_str):
    """Parse timestamp string to microseconds."""
    return int(ts_str)

def resample_books(input_file, output_file, interval_ms=100):
    """
    Resample book CSV to fixed intervals using streaming (memory-efficient).
    Similar to resample_csv.py logic.
    """
    interval_us = interval_ms * 1000  # Convert to microseconds
    
    print(f"Resampling books: {input_file.name}...")
    
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
        print(f"  Error: No valid rows in {input_file}")
        return False
    
    if fieldnames is None:
        print(f"  Error: Could not read fieldnames from {input_file}")
        return False
    
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
    return True

def resample_trades(input_file, output_file, interval_ms=100):
    """
    Resample trades CSV to fixed intervals by aggregating.
    Determines side based on price movement (uptick=buy, downtick=sell).
    If same price, uses higher volume side.
    """
    interval_us = interval_ms * 1000
    
    print(f"Resampling trades: {input_file.name}...")
    
    # Read all trades and group by 100ms intervals
    trades_by_interval = defaultdict(lambda: {'prices': [], 'amounts': [], 'sides': []})
    first_ts = None
    
    with open(input_file, 'r') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        
        prev_price = None
        
        for row in reader:
            try:
                row_ts = parse_timestamp(row['timestamp'])
                price = float(row['price'])
                amount = float(row['amount'])
                side_str = row['side'].lower()
                
                if first_ts is None:
                    first_ts = row_ts
                
                # Determine interval bucket
                interval_start = (row_ts // interval_us) * interval_us
                
                # Store trade in interval
                trades_by_interval[interval_start]['prices'].append(price)
                trades_by_interval[interval_start]['amounts'].append(amount)
                trades_by_interval[interval_start]['sides'].append(side_str)
                
                prev_price = price
                
            except (KeyError, ValueError) as e:
                continue
    
    if first_ts is None:
        print(f"  Error: No valid rows in {input_file}")
        return False
    
    # Generate output intervals from first_ts to last interval
    if not trades_by_interval:
        print(f"  Error: No valid trades found")
        return False
    
    last_interval = max(trades_by_interval.keys())
    current_target_ts = first_ts
    output_count = 0
    
    # Output format: exchange,symbol,timestamp,local_timestamp,id,side,price,amount
    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['exchange', 'symbol', 'timestamp', 'local_timestamp', 'id', 'side', 'price', 'amount'])
        
        while current_target_ts <= last_interval:
            interval_start = (current_target_ts // interval_us) * interval_us
            
            if interval_start in trades_by_interval:
                trades = trades_by_interval[interval_start]
                prices = trades['prices']
                amounts = trades['amounts']
                sides = trades['sides']
                
                # Aggregate: sum amounts
                total_amount = sum(amounts)
                
                # Determine side:
                # 1. If prices are all same, use side with higher volume
                # 2. Otherwise, use price movement (uptick=buy, downtick=sell)
                unique_prices = set(prices)
                
                if len(unique_prices) == 1:
                    # Same price: use side with higher volume
                    buy_volume = sum(amounts[i] for i, s in enumerate(sides) if s == 'buy')
                    sell_volume = sum(amounts[i] for i, s in enumerate(sides) if s == 'sell')
                    side = 'buy' if buy_volume >= sell_volume else 'sell'
                    price = prices[0]
                else:
                    # Price movement: compare first and last price in the interval
                    first_price = prices[0]
                    last_price = prices[-1]
                    
                    # Determine side based on price movement (uptick=buy, downtick=sell)
                    if last_price > first_price:
                        side = 'buy'  # Uptick
                    elif last_price < first_price:
                        side = 'sell'  # Downtick
                    else:
                        # Same start/end but different in between: use volume-weighted side
                        buy_volume = sum(amounts[i] for i, s in enumerate(sides) if s == 'buy')
                        sell_volume = sum(amounts[i] for i, s in enumerate(sides) if s == 'sell')
                        side = 'buy' if buy_volume >= sell_volume else 'sell'
                    
                    # Use volume-weighted average price
                    total_value = sum(p * a for p, a in zip(prices, amounts))
                    price = total_value / total_amount if total_amount > 0 else prices[0]
                
                # Write aggregated trade
                # Use interval_start as the timestamp (aligned to 100ms boundary)
                writer.writerow([
                    'deribit',  # exchange
                    'BTC_USDC-PERPETUAL',  # symbol
                    str(interval_start),  # timestamp (aligned to 100ms boundary)
                    str(interval_start),  # local_timestamp (use same as timestamp)
                    f'AGG-{interval_start}',  # id
                    side,
                    f'{price:.8f}',
                    f'{total_amount:.8f}'
                ])
                output_count += 1
            
            current_target_ts += interval_us
    
    print(f"  Output: {output_count} rows")
    return True

def extract_date_from_filename(filename):
    """Extract date from filename like deribit_book_snapshot_25_2025-10-01_BTC_USDC-PERPETUAL.csv.gz"""
    # Look for YYYY-MM-DD pattern
    import re
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    return None

def main():
    """Main function to process all training data files."""
    base_dir = Path("/home/pravin/dev/alphaqraft_litepool/data/training")
    books_dir = base_dir / "books"
    trades_dir = base_dir / "trades"
    
    # Create temp directories for unzipped files
    temp_books_dir = base_dir / "books_temp"
    temp_trades_dir = base_dir / "trades_temp"
    temp_books_dir.mkdir(exist_ok=True)
    temp_trades_dir.mkdir(exist_ok=True)
    
    # Step 1: Unzip all files
    print("Step 1: Unzipping files...\n")
    book_files = sorted(books_dir.glob("*.gz"))
    trade_files = sorted(trades_dir.glob("*.gz"))
    
    unzipped_books = {}
    unzipped_trades = {}
    
    for gz_file in book_files:
        csv_name = gz_file.stem  # Remove .gz
        output_file = temp_books_dir / csv_name
        print(f"Unzipping: {gz_file.name}")
        with gzip.open(gz_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        date = extract_date_from_filename(csv_name)
        if date:
            unzipped_books[date] = output_file
    
    for gz_file in trade_files:
        csv_name = gz_file.stem
        output_file = temp_trades_dir / csv_name
        print(f"Unzipping: {gz_file.name}")
        with gzip.open(gz_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        date = extract_date_from_filename(csv_name)
        if date:
            unzipped_trades[date] = output_file
    
    print(f"\nUnzipped {len(unzipped_books)} book files and {len(unzipped_trades)} trade files\n")
    
    # Step 2: Resample files
    print("Step 2: Resampling files to 100ms intervals...\n")
    
    resampled_books = {}
    resampled_trades = {}
    
    # Resample books
    for date, input_file in unzipped_books.items():
        output_file = temp_books_dir / f"resampled_{date}.csv"
        if resample_books(input_file, output_file):
            resampled_books[date] = output_file
        print()
    
    # Resample trades
    for date, input_file in unzipped_trades.items():
        output_file = temp_trades_dir / f"resampled_{date}.csv"
        if resample_trades(input_file, output_file):
            resampled_trades[date] = output_file
        print()
    
    # Step 3: Rename files to 1.csv, 2.csv, etc. matching by date
    print("Step 3: Renaming files...\n")
    
    # Get all dates that exist in both books and trades
    common_dates = sorted(set(resampled_books.keys()) & set(resampled_trades.keys()))
    
    if not common_dates:
        print("Error: No matching dates between books and trades!")
        return
    
    # Rename files
    for idx, date in enumerate(common_dates, start=1):
        new_book_name = f"{idx}.csv"
        new_trade_name = f"{idx}.csv"
        
        book_src = resampled_books[date]
        trade_src = resampled_trades[date]
        
        book_dst = books_dir / new_book_name
        trade_dst = trades_dir / new_trade_name
        
        print(f"Renaming {date}:")
        print(f"  Books: {book_src.name} -> {book_dst.name}")
        print(f"  Trades: {trade_src.name} -> {trade_dst.name}")
        
        shutil.move(str(book_src), str(book_dst))
        shutil.move(str(trade_src), str(trade_dst))
    
    # Cleanup temp directories
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_books_dir)
    shutil.rmtree(temp_trades_dir)
    
    print(f"\nDone! Processed {len(common_dates)} date pairs.")
    print(f"Books: {books_dir}")
    print(f"Trades: {trades_dir}")

if __name__ == "__main__":
    main()


#!/bin/bash

# Get all CSV files sorted alphabetically
csv_files=(*.csv)

# Check if there are any CSV files
if [ ${#csv_files[@]} -eq 0 ]; then
    echo "No CSV files found in the current directory."
    exit 1
fi

# Sort the array (optional)
IFS=$'\n' csv_files=($(sort <<<"${csv_files[*]}"))
unset IFS

echo "Renaming CSV files in alphabetical order..."

# Rename files sequentially
counter=1
for file in "${csv_files[@]}"; do
    new_name="${counter}.csv"
    
    # Skip if file already has the correct name
    if [[ "$file" == "$new_name" ]]; then
        echo "  Skipping: $file (already correctly named)"
        ((counter++))
        continue
    fi
    
    # Handle name conflicts
    while [[ -f "$new_name" && "$new_name" != "$file" ]]; do
        echo "  Warning: $new_name already exists, skipping to next number"
        ((counter++))
        new_name="${counter}.csv"
    done
    
    # Rename the file
    mv -- "$file" "$new_name"
    echo "  Renamed: $file -> $new_name"
    
    ((counter++))
done

echo "Done!"

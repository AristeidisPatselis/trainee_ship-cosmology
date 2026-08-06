"""
check_folders.py
================
Check the directory structure and find all model folders and their results.
"""

import os
import glob

BASE_DIR = "/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/Academia/Patra-Physics/Traineeship/Codes_0"

print("=" * 70)
print("DIRECTORY STRUCTURE CHECK")
print("=" * 70)
print(f"Base directory: {BASE_DIR}")
print()

# List all directories
print("All directories in BASE_DIR:")
for item in sorted(os.listdir(BASE_DIR)):
    full_path = os.path.join(BASE_DIR, item)
    if os.path.isdir(full_path):
        print(f"  📁 {item}")
        # Check for result files
        result_files = []
        for root, dirs, files in os.walk(full_path):
            for f in files:
                if f.endswith('.txt') or f.endswith('.npy'):
                    if 'summary' in f.lower() or 'fit' in f.lower() or 'contour' in f.lower():
                        rel_path = os.path.relpath(os.path.join(root, f), full_path)
                        result_files.append(rel_path)
            if len(result_files) > 5:
                break
        
        if result_files:
            print(f"     Found result files:")
            for rf in result_files[:5]:
                print(f"       - {rf}")
            if len(result_files) > 5:
                print(f"       ... and {len(result_files) - 5} more")
        else:
            print(f"     No result files found")

print()
print("=" * 70)

# Find all fit_summary files
print("\nAll fit_summary files found:")
for root, dirs, files in os.walk(BASE_DIR):
    for f in files:
        if 'summary' in f.lower() and f.endswith('.txt'):
            rel_path = os.path.relpath(os.path.join(root, f), BASE_DIR)
            print(f"  - {rel_path}")

print()
print("=" * 70)

# Find all contour files
print("\nAll contour .npy files found:")
for root, dirs, files in os.walk(BASE_DIR):
    for f in files:
        if 'contour' in f.lower() and f.endswith('.npy'):
            rel_path = os.path.relpath(os.path.join(root, f), BASE_DIR)
            print(f"  - {rel_path}")
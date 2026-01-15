"""
Test script to verify error capture functionality.

This script generates various types of errors to test that
they are properly captured in the errors.log file.
"""

import sys
import os

# Initialize error capture
from utils.error_capture import initialize_error_capture

# Initialize error capture
initialize_error_capture("errors.log")

print("Testing error capture...")
print("Errors should appear in terminal AND be logged to errors.log")
print("-" * 60)

# Test 1: Python exception
print("\nTest 1: Python Exception")
try:
    raise ValueError("This is a test ValueError")
except ValueError as e:
    print(f"Caught: {e}", file=sys.stderr)

# Test 2: Import error simulation
print("\nTest 2: Import Error")
try:
    import nonexistent_module_12345
except ImportError as e:
    print(f"Import failed: {e}", file=sys.stderr)

# Test 3: Direct error message
print("\nTest 3: Direct Error Message")
print("ERROR: This is a direct error message", file=sys.stderr)

# Test 4: Traceback
print("\nTest 4: Full Traceback")
def cause_error():
    x = 1 / 0

try:
    cause_error()
except ZeroDivisionError:
    import traceback
    traceback.print_exc()

# Test 5: Non-error message (should NOT be logged)
print("\nTest 5: Non-Error Message (should NOT be in log)")
print("This is just informational output", file=sys.stderr)

print("\n" + "-" * 60)
print("Test complete. Check errors.log file for captured errors.")
print("Only actual errors should be in the log file.")

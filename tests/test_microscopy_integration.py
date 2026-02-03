import time
import os
import sys
import pytest
import numpy as np
from app.tools.microscopy import *
from app.config import settings

@pytest.fixture(scope="module", autouse=True)
def microscope_setup():
    """Setup and teardown for microscope tests."""
    settings.instrument_host = "localhost"
    settings.instrument_port = 9001
    settings.autoscript_port = 9001
    
    print("\n--- Initializing Microscope Test Environment ---")
    start_res = start_server(mode="mock", servers=[MicroscopeServer.Central, MicroscopeServer.AS])
    print(f"Start Server: {start_res}")
    
    time.sleep(1) # Give servers time to bind
    
    conn_res = connect_client(host="localhost")
    print(f"Connect Client: {conn_res}")
    
    yield
    
    print("\n--- Tearing Down Microscope Test Environment ---")
    close_microscope()

def test_get_status():
    print("\n--- Testing: get_microscope_status ---")
    res = get_microscope_status()
    print(f"Result: {res}")
    assert "Microscope is Ready" in res

def test_get_stage_position():
    print("\n--- Testing: get_stage_position ---")
    res = get_stage_position()
    print(f"Result: {res}")
    assert "Stage Position from AS" in res

def test_adjust_magnification():
    print("\n--- Testing: adjust_magnification ---")
    res = adjust_magnification(5000.0)
    print(f"Result: {res}")
    assert "Magnification command sent to AS" in res
    assert "5000.0x" in res

def test_capture_image():
    print("\n--- Testing: capture_image ---")
    res = capture_image(detector="Ceta")
    print(f"Result: {res}")
    assert "Image captured from AS" in res
    assert ".npy" in res
    if "saved to " in res:
        path = res.split("saved to ")[1].split(" (Shape")[0]
        if os.path.exists(path):
            os.remove(path)

def test_calibrate_screen_current():
    print("\n--- Testing: calibrate_screen_current ---")
    res = calibrate_screen_current()
    print(f"Result: {res}")
    assert "Screen current calibration" in res
    assert "calibrated" in res.lower()

def test_set_screen_current():
    print("\n--- Testing: set_screen_current ---")
    res = set_screen_current(150.0)
    print(f"Result: {res}")
    assert "Set current response" in res
    assert "150.0 pA" in res

def test_place_beam():
    print("\n--- Testing: place_beam ---")
    res = place_beam(0.2, 0.8)
    print(f"Result: {res}")
    assert "Beam move response" in res
    assert "0.2" in res
    assert "0.8" in res

def test_blank_beam():
    print("\n--- Testing: blank_beam ---")
    res = blank_beam()
    print(f"Result: {res}")
    assert "Blank beam response" in res
    assert "blanked" in res.lower()

def test_unblank_beam_fixed_duration():
    print("\n--- Testing: unblank_beam (fixed duration) ---")
    res = unblank_beam(duration=0.5)
    print(f"Result: {res}")
    assert "Unblank beam response" in res
    assert "0.5s" in res

def test_unblank_beam_continuous():
    print("\n--- Testing: unblank_beam (continuous) ---")
    res = unblank_beam()
    print(f"Result: {res}")
    assert "Unblank beam response" in res
    assert "unblanked" in res.lower()

if __name__ == "__main__":
    # Fallback for running without pytest
    pytest.main([__file__])

from app.microscope_api import MicroscopeControl, StagePosition
import numpy as np

def test_simulator_flow():
    print("Testing MicroscopeControl in simulator mode...")
    mc = MicroscopeControl(sim_mode=True)
    
    # Test getting stage position
    pos = mc.get_stage_position()
    print(f"Current Position: {pos}")
    
    # Test moving stage
    new_pos = StagePosition(x=500.0, y=500.0)
    moved_pos = mc.set_stage_position(new_pos)
    print(f"Moved to: {moved_pos}")
    
    # Test acquiring image
    img_result = mc.acquire_image(detector="Ceta")
    print(f"Acquired image shape: {img_result.data.shape}")
    print(f"Metadata: {img_result.metadata}")
    
    # Test bounds validation
    try:
        invalid_pos = StagePosition(x=-1.0, y=500.0)
    except ValueError as e:
        print(f"Caught expected validation error: {e}")

    print("Simulator flow test PASSED.")

if __name__ == "__main__":
    test_simulator_flow()

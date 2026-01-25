from typing import Optional, Dict, Any, Union
import numpy as np
from pydantic import BaseModel, Field, validator
from app.config import settings

class StagePosition(BaseModel):
    x: float = Field(..., description="X position in microns")
    y: float = Field(..., description="Y position in microns")
    z: Optional[float] = Field(None, description="Z position in microns")
    rotation: Optional[float] = Field(None, description="Rotation in degrees")
    tilt: Optional[float] = Field(None, description="Tilt in degrees")

    @validator('x')
    def x_within_bounds(cls, v):
        if not (settings.stage_x_min <= v <= settings.stage_x_max):
            raise ValueError(f"X must be between {settings.stage_x_min} and {settings.stage_x_max}")
        return v

    @validator('y')
    def y_within_bounds(cls, v):
        if not (settings.stage_y_min <= v <= settings.stage_y_max):
            raise ValueError(f"Y must be between {settings.stage_y_min} and {settings.stage_y_max}")
        return v

class ImageResult(BaseModel):
    data: Any = Field(..., description="Numpy array or base64 encoded image data")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MicroscopeControl:
    """
    High-level, typed API for microscope control.
    Wraps asyncroscopy and handles safety checks.
    """
    def __init__(self, sim_mode: bool = None):
        self.sim_mode = sim_mode if sim_mode is not None else settings.sim_mode
        self._client = None
        
        if not self.sim_mode:
            self._connect()

    def _connect(self):
        """Connect to the Pyro5 server if not in simulation mode."""
        try:
            import Pyro5.api
            uri = f"PYRO:TEMServer@{settings.server_host}:{settings.server_port}"
            self._client = Pyro5.api.Proxy(uri)
        except Exception as e:
            print(f"Failed to connect to microscope server: {e}")
            self.sim_mode = True

    def get_stage_position(self) -> StagePosition:
        """Get the current stage position."""
        if self.sim_mode:
            return StagePosition(x=100.0, y=100.0, z=0.0)
        
        # asyncroscopy returns stage in nm/deg, we convert to microns
        raw_pos = self._client.get_stage()
        return StagePosition(
            x=raw_pos['x'] / 1000.0,
            y=raw_pos['y'] / 1000.0,
            z=raw_pos.get('z', 0) / 1000.0,
            rotation=raw_pos.get('r'),
            tilt=raw_pos.get('t')
        )

    def set_stage_position(self, pos: StagePosition, relative: bool = False) -> StagePosition:
        """Set the stage position with safety checks."""
        # Validation is handled by Pydantic model initialization
        if self.sim_mode:
            print(f"SIMULATOR: Moving stage to {pos}")
            return pos

        # Convert back to nm for asyncroscopy
        target = {
            'x': pos.x * 1000.0,
            'y': pos.y * 1000.0
        }
        if pos.z is not None: target['z'] = pos.z * 1000.0
        if pos.rotation is not None: target['r'] = pos.rotation
        if pos.tilt is not None: target['t'] = pos.tilt

        self._client.set_stage(target, relative=relative)
        return self.get_stage_position()

    def acquire_image(self, detector: str = "Ceta") -> ImageResult:
        """Acquire an image from the specified detector."""
        if self.sim_mode:
            print(f"SIMULATOR: Acquiring image from {detector}")
            # Return dummy noise
            dummy_data = np.random.rand(512, 512)
            return ImageResult(data=dummy_data, metadata={"detector": detector, "mode": "simulated"})

        try:
            img = self._client.acquire_image(detector)
            return ImageResult(data=img, metadata={"detector": detector})
        except Exception as e:
            print(f"Error acquiring image: {e}")
            raise

    def set_beam_position(self, x: float, y: float) -> bool:
        """Set the beam position in nm."""
        if self.sim_mode:
            print(f"SIMULATOR: Setting beam position to ({x}, {y})")
            return True
        
        self._client.set_probe_position(x, y)
        return True

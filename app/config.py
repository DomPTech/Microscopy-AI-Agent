import os
from typing import Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class MicroscopeSettings(BaseSettings):
    """
    Centralized configuration for microscope hardware and paths.
    Uses environment variables with prefixes (e.g. MICROSCOPE_STAGE_X_MIN).
    """
    model_config = SettingsConfigDict(env_prefix='MICROSCOPE_', env_file='.env', extra='ignore')

    # Hardware Bounds
    stage_x_min: float = Field(0.0, description="Minimum X coordinate for the stage in microns")
    stage_x_max: float = Field(100000.0, description="Maximum X coordinate for the stage in microns")
    stage_y_min: float = Field(0.0, description="Minimum Y coordinate for the stage in microns")
    stage_y_max: float = Field(100000.0, description="Maximum Y coordinate for the stage in microns")
    
    # Image Settings
    max_image_size: int = Field(4096, description="Maximum width/height for image acquisition")
    
    # Paths and Networks
    server_host: str = Field("127.0.0.1", description="Hostname for the asyncroscopy server")
    server_port: int = Field(9093, description="Port for the asyncroscopy server")
    
    # Simulation Mode
    sim_mode: bool = Field(True, description="Enable dry-run/simulator mode by default")

# Global configuration instance
settings = MicroscopeSettings()

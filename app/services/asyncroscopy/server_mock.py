import numpy as np
from twisted.internet import reactor, protocol
import json
import struct

from app.services.asyncroscopy.protocols.execution_protocol import ExecutionProtocol
from app.services.asyncroscopy.protocols.utils import package_message

class MockFactory(protocol.Factory):
    def __init__(self):
        self.status = "Offline"
        # Mock state
        self.magnification = 1000.0
        self.stage_position = np.array([0.0, 0.0, 0.0])

    def buildProtocol(self, addr):
        proto = MockProtocol()
        proto.factory = self
        return proto

class MockProtocol(ExecutionProtocol):
    def __init__(self):
        super().__init__()
        # No register_command needed, getattr handles dispatch.
        
    # Standard commands
    def connect_AS(self, args: dict):
        print(f"[Mock] connect_AS called with {args}")
        self.factory.status = "Ready"
        msg = f"[Mock] Connected to microscope."
        return package_message(msg)

    def get_scanned_image(self, args: dict):
        self.factory.status = "Busy"
        size = int(args.get("size", 512))
        dwell_time = float(args.get("dwell_time", 1e-6))
        
        print(f"[Mock] Generating image size={size}, dwell={dwell_time}")
        
        # Generate noise image
        arr = np.random.randint(0, 255, (size, size), dtype=np.uint8)
        self.factory.status = "Ready"
        
        return package_message(arr)

    def get_stage(self, args: dict = None):
        return package_message(self.factory.stage_position)

    def get_status(self, args: dict = None):
        return package_message(f"Microscope is {self.factory.status}")

    def set_magnification(self, args: dict):
        mag = float(args.get("mag", 1000.0))
        self.factory.magnification = mag
        return package_message(f"Magnification set to {mag}")

    def move_stage(self, args: dict):
        x = float(args.get("x", 0.0))
        y = float(args.get("y", 0.0))
        self.factory.stage_position[0] = x
        self.factory.stage_position[1] = y
        return package_message(f"Stage moved to {x}, {y}")

    # Aliases for direct NotebookClient connection (uses AS_ prefix)
    AS_connect_AS = connect_AS
    AS_get_scanned_image = get_scanned_image
    AS_get_stage = get_stage
    AS_get_status = get_status
    AS_set_magnification = set_magnification
    AS_move_stage = move_stage

if __name__ == "__main__":
    import sys
    port = 9001
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    # Force stdout flush for debugging
    print(f"[Mock] Server running on port {port}...", flush=True)
    reactor.listenTCP(port, MockFactory())
    reactor.run()
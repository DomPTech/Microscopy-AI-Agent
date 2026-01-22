import Pyro5.api
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MockServer")

# For serialization (list, shape, dtype) tuple as used in base_proxy
def serialize(array):
    array_list = array.tolist()
    dtype = str(array.dtype)
    return array_list, array.shape, dtype

@Pyro5.api.expose
class TEMServer(object):
    def __init__(self):
        self.detectors = {
            'wobbler_camera': {'size': 512, 'exposure': 0.1},
            'ceta_camera': {'size': 512, 'exposure': 0.1}
        }
        self.magnification = 1000.0
        self.stage_position = [0.0, 0.0, 0.0, 0.0, 0.0]

    def get_instrument_status(self, parameters=None):
        logger.info(f"get_instrument_status params={parameters}")
        return {
            'vacuum': 'Ready',
            'column_valve': 'Open',
            'beam_current': 1.0e-9
        }

    def get_stage(self):
        logger.info("get_stage")
        return self.stage_position

    def set_stage(self, stage_positions, relative=True):
        logger.info(f"set_stage pos={stage_positions} rel={relative}")
        # Simplistic update map
        mapping = {'x':0, 'y':1, 'z':2, 'a':3, 'b':4}
        for k, v in stage_positions.items():
            if k in mapping:
                idx = mapping[k]
                if relative:
                    self.stage_position[idx] += v
                else:
                    self.stage_position[idx] = v
        return 1

    def acquire_image(self, device_name, **args):
        logger.info(f"acquire_image device={device_name} args={args}")
        # Return random noise
        size = 512
        if device_name in self.detectors:
            size = self.detectors[device_name].get('size', 512)
        
        arr = np.random.randint(0, 255, (size, size), dtype=np.uint16)
        return serialize(arr)
        
    def set_microscope_status(self, parameter=None, value=None):
        logger.info(f"set_microscope_status {parameter}={value}")
        if parameter == 'magnification':
            self.magnification = float(value)
        return 1

    def get_detectors(self):
        return list(self.detectors.keys())

    def close(self):
        logger.info("Closing server")
        # In Pyro, closing object doesn't stop daemon, but we can simulate shutdown if needed
        # Or just return 1
        return 1

def main(host="localhost", port=9093):
    print(f"Mock Pyro5 Server running on {host}:{port}")
    daemon = Pyro5.api.Daemon(host=host, port=port)
    uri = daemon.register(TEMServer, objectId="tem.server")
    print("Server ready. URI:", uri)
    daemon.requestLoop()

if __name__ == "__main__":
    import sys
    port = 9093
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    main(port=port)

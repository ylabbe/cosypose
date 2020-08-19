import os
import subprocess
import xml.etree.ElementTree as ET
import pkgutil
import pybullet as pb
from .client import BulletClient


class BaseScene:
    _client_id = -1
    _client = None
    _connected = False
    _simulation_step = 1/240.

    def connect(self, gpu_renderer=True, gui=False):
        assert not self._connected, 'Already connected'
        if gui:
            self._client_id = pb.connect(pb.GUI, '--width=640 --height=480')
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1, physicsClientId=self._client_id)
            pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1, physicsClientId=self._client_id)
            pb.configureDebugVisualizer(pb.COV_ENABLE_TINY_RENDERER, 0, physicsClientId=self._client_id)
        else:
            self._client_id = pb.connect(pb.DIRECT)
        if self._client_id < 0:
            raise Exception('Cannot connect to pybullet')
        if gpu_renderer and not gui:
            os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
            os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
            # Get EGL device
            assert 'CUDA_VISIBLE_DEVICES' in os.environ
            devices = os.environ.get('CUDA_VISIBLE_DEVICES', ).split(',')
            assert len(devices) == 1
            out = subprocess.check_output(['nvidia-smi', '--id='+str(devices[0]), '-q', '--xml-format'])
            tree = ET.fromstring(out)
            gpu = tree.findall('gpu')[0]
            dev_id = gpu.find('minor_number').text
            os.environ['EGL_VISIBLE_DEVICES'] = str(dev_id)
            egl = pkgutil.get_loader('eglRenderer')
            pb.loadPlugin(egl.get_filename(), "_eglRendererPlugin", physicsClientId=self._client_id)
        pb.resetSimulation(physicsClientId=self._client_id)
        self._connected = True
        self._client = BulletClient(self._client_id)

        self.client.setPhysicsEngineParameter(numSolverIterations=50)
        self.client.setPhysicsEngineParameter(fixedTimeStep=self._simulation_step)
        self.client.setGravity(0, 0, -9.8)

    def run_simulation(self, time):
        n_steps = float(time) / self._simulation_step
        for _ in range(int(n_steps)):
            self.client.stepSimulation()

    def disconnect(self):
        try:
            pb.resetSimulation(physicsClientId=self._client_id)
            pb.disconnect(physicsClientId=self._client_id)
        except pb.error:
            pass
        self._connected = False
        self._client_id = -1

    @property
    def client(self):
        assert self._connected
        return self._client

    @property
    def client_id(self):
        assert self._connected
        return self._client_id

    def __del__(self):
        self.disconnect()

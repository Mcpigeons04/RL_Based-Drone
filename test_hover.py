import numpy as np
from stable_baselines3 import PPO
import time
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel

class CustomHover(HoverAviary):
    def _computeReward(self):
        state = self._getDroneStateVector(0)

        # position
        pos = state[0:3]
        target = np.array([0, 0, 1])

        # velocity
        vel = state[10:13]
        v = np.linalg.norm(vel)

        # reward logic for our model
        return -np.linalg.norm(pos - target) - 0.3 * v
    

env = CustomHover(
    drone_model=DroneModel.CF2X,
    initial_xyzs=np.array([[0, 0, 1]]),
    physics='pyb',
    gui=True
)

model = PPO.load("hover_model_700k_vel")

obs, info = env.reset()

for i in range(2000):
    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    time.sleep(1/240)

    if done:
        obs, info = env.reset()


input("Press enter to close")

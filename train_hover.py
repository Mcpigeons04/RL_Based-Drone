import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel

class CustomHover(HoverAviary):

    def _computeReward(self):
        state= self._getDroneStateVector(0)
        #position
        pos= state[0:3]
        target= np.array([0,0,1])
        #velocity
        vel= state[10:13]
        v= np.linalg.norm(vel)
        #reward logic for our model
        return -np.linalg.norm(pos-target) - 0.3*v


def make_env():
    return CustomHover(
        drone_model=DroneModel.CF2X,
        initial_xyzs=np.array([[0, 0, 1]]),
        physics='pyb'
    )


env = make_vec_env(make_env, n_envs=1)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4
)

model.learn(total_timesteps=700000, log_interval=10)

model.save("hover_model_700k_vel")

print("Training complete!")





from gym import Wrapper
from utils import plot_to_array
from PIL import Image
from einops import rearrange

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.integrate import odeint


class Planets(object):
    """
    Implements a 2D environments where there are N bodies (planets) that attract each other according to a 1/r law.

    We assume the mass of each body is 1.
    """

    # For each dimension of the hypercube
    MIN_POS = 0.  # if box exists
    MAX_POS = 1.  # if box exists
    INIT_MAX_VEL = 1.
    GRAVITATIONAL_CONSTANT = 1.

    def __init__(self, num_bodies, num_dimensions=2, dt=0.01, contained_in_a_box=True):
        self.num_bodies = num_bodies
        self.num_dimensions = num_dimensions
        self.dt = dt
        self.contained_in_a_box = contained_in_a_box

        # state variables
        self.body_positions = None
        self.body_velocities = None
        
        self.body_colors = np.random.uniform(size=self.num_bodies)

        self.reset()

    def reset(self):
        self.body_positions = np.random.uniform(self.MIN_POS, self.MAX_POS, size=(self.num_bodies, self.num_dimensions))
        self.body_velocities = self.INIT_MAX_VEL * np.random.uniform(-1, 1, size=(self.num_bodies, self.num_dimensions))

    @property
    def state(self):
        return np.concatenate((self.body_positions, self.body_velocities), axis=1)  # (N, 2D)

    def step(self):

        # Helper functions since ode solver requires flattened inputs
        def flatten(positions, velocities):  # positions shape (N, D); velocities shape (N, D)
            system_state = np.concatenate((positions, velocities), axis=1)  # (N, 2D)
            system_state_flat = system_state.flatten()  # ode solver requires flat, (N*2D,)
            return system_state_flat

        def unflatten(system_state_flat):  # system_state_flat shape (N*2*D,)
            system_state = system_state_flat.reshape(self.num_bodies, 2 * self.num_dimensions)  # (N, 2*D)
            positions = system_state[:, :self.num_dimensions]  # (N, D)
            velocities = system_state[:, self.num_dimensions:]  # (N, D)
            return positions, velocities

        # ODE function
        def system_first_order_ode(system_state_flat, _):

            positions, velocities = unflatten(system_state_flat)
            accelerations = np.zeros_like(velocities)  # init (N, D)

            for i in range(self.num_bodies):
                relative_positions = positions - positions[i]  # (N, D)
                distances = np.linalg.norm(relative_positions, axis=1, keepdims=True)  # (N, 1)
                distances[i] = 1.  # bodies don't affect themselves, and we don't want to divide by zero next

                # forces (see https://en.wikipedia.org/wiki/Numerical_model_of_the_Solar_System)
                force_vectors = self.GRAVITATIONAL_CONSTANT * relative_positions / (distances**self.num_dimensions)  # (N,D)
                force_vector = np.sum(force_vectors, axis=0)  # (D,)
                accelerations[i] = force_vector  # assuming mass 1.

            d_system_state_flat = flatten(velocities, accelerations)
            return d_system_state_flat

        # integrate + update
        current_system_state_flat = flatten(self.body_positions, self.body_velocities)  # (N*2*D,)
        _, next_system_state_flat = odeint(system_first_order_ode, current_system_state_flat, [0., self.dt])  # (N*2*D,)
        self.body_positions, self.body_velocities = unflatten(next_system_state_flat)  # (N, D), (N, D)

        # bounce off boundaries of box
        if self.contained_in_a_box:
            ind_below_min = self.body_positions < self.MIN_POS
            ind_above_max = self.body_positions > self.MAX_POS
            self.body_positions[ind_below_min] += 2. * (self.MIN_POS - self.body_positions[ind_below_min])
            self.body_positions[ind_above_max] += 2. * (self.MAX_POS - self.body_positions[ind_above_max])
            self.body_velocities[ind_below_min] *= -1.
            self.body_velocities[ind_above_max] *= -1.
            self.assert_bodies_in_box()  # check for bugs

    def render_rgbarray(self):
        x = self.body_positions[:,0]
        y = self.body_positions[:,1]

        fig, ax = plt.subplots(1,1, figsize=(5,5))
        ax.scatter(x, y, marker="*", c=self.body_colors, cmap='viridis', s=1_000)
        ax.set_xlim(self.MIN_POS, self.MAX_POS)
        ax.set_ylim(self.MIN_POS, self.MAX_POS)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        # ax.axis('off')
        ax.set_facecolor('black')
        ax.margins(x=0., y=0.)
        fig.tight_layout()
        img = plot_to_array(fig) # (H,W,C)
        plt.close(fig)
        return img

    def assert_bodies_in_box(self):
        """
        if the sim goes really fast, they can bounce one-step out of box. Let's just check for this for now, fix later
        """
        assert np.all(self.body_positions >= self.MIN_POS) and np.all(self.body_positions <= self.MAX_POS)

    @property
    def temperature(self):
        """
        Temperature is the average kinetic energy of system
        :return: float
        """
        average_kinetic_energy = 0.5 * np.mean(np.linalg.norm(self.body_velocities, axis=1))  # (N, D) --> (1,)
        return average_kinetic_energy


class Electrons(Planets):
    """
    Implements a 2D environments where there are N bodies (electrons) that repel each other according to a 1/r law.
    """

    # override
    GRAVITATIONAL_CONSTANT = -1.  # negative means they repel


class IdealGas(Planets):
    """
    Implements a 2D environments where there are N bodies (gas molecules) that do not interact with each other.
    """

    # override
    GRAVITATIONAL_CONSTANT = 0.  # zero means they don't interact




class DistractorWrapper(Wrapper):

    def __init__(self, env, distractor_kwargs):
        self.env = env
        self.distractor = IdealGas(**distractor_kwargs)

    def reset(self, seed=None, options=None):
        # Reset the distrator env
        self.distractor.reset()
        try:
            return self.env.reset(seed, options)
        except TypeError:
            return self.env.reset()
    
    def step(self, action):
        # Step the distractor env
        self.distractor.step()
        _, rew, info, terminated, truncated =  self.env.step(action)

        obs_distract = self.render()
        obs_distract = rearrange(obs_distract, 'h w c -> c h w')
        return obs_distract, rew, info, terminated, truncated
        
    
    def render(self, **kwargs):

        # Foreground image from base env
        try:
            env_img = self.env.render_rgbarray()
        except: 
            env_img = self.env.render()

        env_img = Image.fromarray(env_img).convert('RGBA')


        # Background image from distractor
        distractor_img = self.distractor.render_rgbarray()
        distractor_img = Image.fromarray(distractor_img).convert('RGBA')
        distractor_img = distractor_img.resize((env_img.width, env_img.height))


        overlay_img = Image.blend(env_img, distractor_img, 0.4).convert('RGB')
        
        return np.array(overlay_img)
        
if __name__ == "__main__":

    from gridworld.gridworld import GridWorldRGB
    from reacher.reacher import MOReacherRGB

    base_env = GridWorldRGB({
        'size': 20,
        'goals': [[10,10]],
        'goal_weights': [1],
        'obstacles': [[5,5]],
        'obstacle_weights': [1]
    })
    
    # base_env = MOReacherRGB({})

    distractor_kwargs = {
        'num_bodies': 10,
        'num_dimensions': 2,
    }
    env = DistractorWrapper(base_env, distractor_kwargs)

    env.reset()
    imgs = []
    for idx in range(20):
        obs = env.step(env.action_space.sample())
        imgs.append(env.render())
    imgs = [Image.fromarray(x) for x in imgs]

    # Save the list of images as a GIF
    imgs[0].save(
        "distractors.gif",
        save_all=True,
        append_images=imgs[1:],
        duration=200,  # Duration in milliseconds between frames
        loop=0  # 0 means an infinite loop
    )

    print("Done")


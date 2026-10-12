import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt


# three types of disturbances corresponding to different places where uncertainty can enter the system
# observation disturbance (sensor noise), 
# action disturbance (controller asks for 1.2 torque, but accidentally get 1.25 torque),
# and state disturbance (wind, unexpected external conditions)
class Disturbance:
    def __init__(self, xo, xa, xs):
        self.xo = xo
        self.xa = xa
        self.xs = xs
        pass


# This class stores the probability distributions used to generate those disturbances: 
# how likely different errors are; like xa distribution can distributed as normal(0 (mean), 0.05^2 (var))
class DisturbanceDistribution:
    def __init__(self, Do, Da, Ds):
        self.Do = Do
        self.Da = Da
        self.Ds = Ds
        pass


def plot_observation_disturbances(xs, sigma):
    fig, ax = plt.subplots()
    for x in xs:
        ax.scatter(x.xo[0, 0], x.xo[1, 0], color="orange")
    ax.set_xlabel(r"xo, $\theta$")
    ax.set_ylabel(r"xo, $\omega$")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_title(r"$\sigma = $ " + f"{sigma}")
    plt.savefig(f"../../results/disturbance_plot_{sigma}.svg", dpi=300)
    return ax


def plot_trajectories(theta_matrix, sigma):
    fig, ax = plt.subplots()
    ax.plot(theta_matrix, color="steelblue", alpha=0.5, linewidth=0.8)
    plt.axhline(np.pi / 4, color="red", linewidth=1)
    plt.axhline(-np.pi / 4, color="red", linewidth=1)
    ax.set_title(r"$\sigma = $ " + f"{sigma}")
    plt.xlabel("t, [s]")
    plt.ylabel(r"$\theta$, [rad]")
    plt.savefig(f"../../results/inverted_pendulum_{sigma}.svg", dpi=300)
    return ax


# the goal is for robot to move towards position 5
# the state of this robot is the distance from starting pos in feet (int)
# we observe this using a ruler
# we use this observation and conduct action: if > 5 move left, if <= 5, move right
# we use this action and our true state to calc our next state

class oneD_robot:

    def get_observation(self, s, xo):
        return s + xo

    def get_action(self, o, xa):
        if o < 5.01 and o > 4.99:
            a = xa
        elif o > 5:
            a = -1 + xa
        else:
            a = 1 + xa

        return a

    def get_state(self, s, a, xs):
        pos = s
        pos = s + a

        return pos + xs

    def step(self, s, disturbance_dist):
        xo = disturbance_dist.Do.rvs()
        o = self.get_observation(s, xo)

        xa = disturbance_dist.Da.rvs()
        a = self.get_action(o, xa)

        xs = disturbance_dist.Ds.rvs()
        sprime = self.get_state(s, a, xs)

        return o, a, sprime, Disturbance(xo, xa, xs)

    # robot must stay within positions -7 to 7
    def check_spec(self, trajectory):
        return (np.abs(trajectory) <= 7).all()

    def rollout(self, n_steps, disturbance_dist):
        # Start the robot at position 0
        s = 0

        # Store the robot's position at each step
        positions = np.zeros(n_steps)
        positions[0] = s

        # Store the disturbances that occurred
        xs = []

        # Run the robot one step at a time
        for i in range(1, n_steps):
            o, a, s, x = self.step(s, disturbance_dist)

            positions[i] = s
            xs.append(x)

        return positions, xs

# we built a system that we're going to falsify


if __name__ == "__main__":
    n_rollouts = 300
    sig_o = 1.0
    sig_a = 0.2
    sig_s = 1.0

    # makes sense observation has disturbance beeause sensor measured position incorrectly
    # makes sense action has disturbance b/c robot didn't execute movement exactly
    # makes sense state has disturbance b/c maybe we conducted this outside so external factors like wind can push robot
    disturbance_dist = DisturbanceDistribution(
        Do=norm(0, sig_o), Da=norm(0, sig_a), Ds=norm(0, sig_s)
    )
    
    n_steps = 20
    robot = oneD_robot()

    # Stores whether each rollout passed the safety specification
    successes = np.zeros(n_rollouts)

    # Stores all robot position trajectories
    position_matrix = np.zeros((n_steps, n_rollouts))

    for r in range(n_rollouts):
        positions, xs = robot.rollout(n_steps, disturbance_dist)

        position_matrix[:, r] = positions

        successes[r] = robot.check_spec(positions)

    print(f"Success Rate: {successes.sum() / n_rollouts}")






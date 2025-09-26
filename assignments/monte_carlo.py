import numpy as np
import gymnasium as gym
from typing import Iterable, Tuple

from interfaces.policy import Policy

def off_policy_mc_prediction_weighted_importance_sampling(
    observation_space: gym.spaces.Discrete,
    action_space: gym.spaces.Discrete,
    trajs: Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi: Policy,
    pi: Policy,
    initQ: np.ndarray,
    gamma: float = 1.0
) -> np.ndarray:
    """
    Evaluate the estimated Q values of the target policy using off-policy Monte-Carlo prediction algorithm
    with *weighted* importance sampling. 

    The algorithm can be found in Sutton & Barto 2nd edition p. 110.

    Every-visit implementation is fine.

    Parameters:
        env_spec (EnvSpec): environment spec
        trajs (list): A list of N trajectories generated using behavior policy bpi
            - Each element is a tuple representing (s_t, a_t, r_{t+1}, s_{t+1})
        bpi (Policy): behavior policy used to generate trajectories
        pi (Policy): evaluation target policy
        initQ (np.ndarray): initial Q values; np array shape of [nS, nA]

    Returns:
        Q (np.ndarray): $q_pi$ function; numpy array shape of [nS, nA]
    """
    nS: int = observation_space.n
    """The number of states in the environment."""
    nA: int = action_space.n
    """The discount factor."""
    Q: np.ndarray = initQ.copy()
    """The Q(s, a) function to estimate."""
    C: np.ndarray = np.zeros((nS, nA), dtype=np.float64)
    """The importance sampling ratios."""


    for traj in trajs:
        episode = list(traj)
        G = 0.0
        W = 1.0

        # Process trajectory in reverse order
        for i in reversed(range(len(episode))):
            s, a, r, s_next = episode[i]
            G = gamma * G + r
            C[s, a] += W
            if C[s, a] > 0.0:
                Q[s, a] += (W / C[s, a]) * (G - Q[s, a])

            pi_prob = pi.action_prob(s, a)
            bpi_prob = bpi.action_prob(s, a)
            if bpi_prob == 0:
                break

            W *= pi_prob / bpi_prob
            if W == 0.0:
                break

    return Q
        
def off_policy_mc_prediction_ordinary_importance_sampling(
    observation_space: gym.spaces.Discrete,
    action_space: gym.spaces.Discrete,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array,
    gamma:float = 1.0
) -> np.array:
    """
    Evaluate the estimated Q values of the target policy using off-policy Monte-Carlo prediction algorithm
    with *ordinary* importance sampling. 

    The algorithm with weighted importance sampling can be found in Sutton & Barto 2nd edition p. 110.
    You will need to make a small adjustment for ordinary importance sampling.

    Carefully look at page 109.

    Every-visit implementation is fine.

    Parameters:
        env_spec (EnvSpec): environment spec
        trajs (list): A list of N trajectories generated using behavior policy bpi
            - Each element is a tuple representing (s_t, a_t, r_{t+1}, s_{t+1})
        bpi (Policy): behavior policy used to generate trajectories
        pi (Policy): evaluation target policy
        initQ (np.ndarray): initial Q values; np array shape of [nS, nA]
        
    Returns:
        Q (np.ndarray): $q_pi$ function; numpy array shape of [nS, nA]
    """
    nS: int = observation_space.n
    """The number of states in the environment."""
    nA: int = action_space.n
    """The number of actions in the environment."""
    Q: np.ndarray = initQ.copy()
    """The Q(s, a) function to estimate."""
    C: np.ndarray = np.zeros((nS, nA), dtype=np.float64)
    """The importance sampling ratios."""

    for traj in trajs:
        episode = list(traj)
        G = 0.0
        W = 1.0

        for i in reversed(range(len(episode))):
            s, a, r, s_next = episode[i]
            G = gamma * G + r
            
            pi_prob = pi.action_prob(s, a)            
            bpi_prob = bpi.action_prob(s, a)
            
            if bpi_prob == 0.0:
                break
                
            C[s, a] += 1
            Q[s, a] += W * G - Q[s, a]) / C[s, a]

            W *= pi_prob / bpi_prob
            if W == 0.0:
                break

    return Q
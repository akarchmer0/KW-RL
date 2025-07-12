# kw_and_fourier_env.py
import numpy as np
import random
import gymnasium as gym
from typing import Optional
import itertools
import math

def bool_fourier_coeffs(f_vec: np.ndarray, n: int, max_level: int = 3):
    """Return exact Fourier coefficients up to max_level."""
    N = 1 << n
    assert len(f_vec) == N

    # Pre-compute all χ_S(x) for S up to max_level
    x_bits = np.arange(N)[:, None] >> np.arange(n) & 1  # (N,n) matrix of bits
    coeffs = []

    for level in range(max_level + 1):
        for S in itertools.combinations(range(n), level):
            mask = np.array(S, dtype=np.intp)           # indices in S
            parity = x_bits[:, mask].sum(axis=1) & 1    # Σ_{i∈S} x_i mod 2
            chi = 1 - 2 * parity                        # (-1)^{parity}
            coeff = np.dot(f_vec, chi) / N              # exact mean
            coeffs.append(float(coeff))
    return np.array(coeffs, dtype=np.float32)


class KWAndFourierEnv(gym.Env):
    """
    Two-player cooperative KW game for an *arbitrary* Boolean function f.
    Player-0 gets f and x s.t. f(x)=1
    Player-1 gets f and y s.t. f(y)=0
    Goal: find i with x_i != y_i.
    The players receive the Boolean Fourier spectrum of f (up to max_level)
    concatenated to their private string.
    """

    def __init__(self, n: int = 4, max_fourier_level = None):
        super().__init__()
        self.n = n
        self.max_len = 2 * n + 1
        self.total_steps = 0

        self.max_fourier_level = n
        if max_fourier_level:
            self.max_fourier_level = max_fourier_level
        
        # ---- action spaces identical to KWAndEnv ----
        self.action_space_p0 = gym.spaces.Discrete(3 + n)
        self.action_space_p1 = gym.spaces.Discrete(3)
        

        # ---- compute Fourier feature size ----
        self.fourier_size = sum(math.comb(n, l) for l in range(self.max_fourier_level + 1))
        history_size = self.max_len * 3
        obs_size = n + self.fourier_size + history_size
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.reset()

    @property
    def obs_size(self):
        return self.n + self.fourier_size + self.max_len * 3

    # ------------------------------------------------------------
    # reset & helper utilities
    # ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.done = False

        # 1. Sample a random Boolean function f:{0,1}^n → {0,1}
        N = 1 << self.n
        self.f_vec = np.random.randint(0, 2, size=N, dtype=np.uint8)

        # 2. Ensure f is non-constant (otherwise KW is trivial)
        if np.all(self.f_vec == 0) or np.all(self.f_vec == 1):
            self.f_vec[0] ^= 1  # flip one bit

        # 3. Compute Fourier coefficients once
        self.fourier = bool_fourier_coeffs(self.f_vec, self.n, self.max_fourier_level)

        # 4. Sample inputs x, y respecting f
        ones = np.where(self.f_vec == 1)[0]
        zeros = np.where(self.f_vec == 0)[0]
        x_idx = np.random.choice(ones)
        y_idx = np.random.choice(zeros)
        self.x = np.array([(x_idx >> i) & 1 for i in range(self.n)], dtype=np.int8)
        self.y = np.array([(y_idx >> i) & 1 for i in range(self.n)], dtype=np.int8)

        # 5. Where do they differ?
        self.diff_idx = np.where(self.x != self.y)[0]

        # bookkeeping
        self.current_turn = 0
        self.last_action = {0: None, 1: None}
        self.history = np.zeros((self.max_len, 3), dtype=np.float32)
        return self._obs(0), self._obs(1)

    # ------------------------------------------------------------
    # observation encoding
    # ------------------------------------------------------------
    def _obs(self, player):
        bits = self.x if player == 0 else self.y
        flat_hist = self.history.reshape(-1)
        return np.concatenate([bits.astype(np.float32),
                               self.fourier,
                               flat_hist])

    # ------------------------------------------------------------
    # all action / step / reward logic identical to KWAndEnv
    # ------------------------------------------------------------
    # ... (copy _can_pass, _is_pass_action, _is_bit_action, _is_decision_action, step verbatim)
    # ------------------------------------------------------------

    def _can_pass(self, player):
        last_action = self.last_action[player]
        if last_action is None:
            # Allow first turn passes, but only for P0 to start the conversation
            return player == 0 and self.t == 0
        return self._is_bit_action(last_action)

    def _is_pass_action(self, action, player):
        """Check if action is a pass for the given player"""
        if player == 0:
            return action == (2 + self.n)  # Last action in P0's space
        else:  # player == 1
            return action == 2  # Last action in P1's space

    def _is_bit_action(self, action):
        """Check if action is sending a bit (0 or 1)"""
        return action in [0, 1]

    def _is_decision_action(self, action, player):
        """Check if action is a decision for the given player"""
        if player == 0:
            return 2 <= action < (2 + self.n)  # Decision range for P0
        else:  # player == 1
            return False  # P1 cannot make decisions

    def step(self, actions: tuple[int, int]):
        """
        MODIFIED: Process actions in turns instead of simultaneously
        actions[0] = P0's intended action
        actions[1] = P1's intended action
        Only current_turn player's action is executed
        """
        rewards = [0.0, 0.0]

        # TURN-BASED: Only execute current player's action
        if self.current_turn == 0:
            active_action = actions[0]  # P0's turn
            active_player = 0
        else:
            active_action = actions[1]  # P1's turn
            active_player = 1

        # Validate action bounds
        max_action = (3 + self.n) if active_player == 0 else 3
        if not (0 <= active_action < max_action):
            rewards = [-1.0, -1.0]
            self.done = True
            return (self._obs(0), self._obs(1)), rewards, self.done, False, {}

        # check if pass action is allowed
        if self._is_pass_action(active_action, active_player):
            if not self._can_pass(active_player):
                # ILLEGAL PASS: Player tried to pass without sending a bit last turn
                rewards = [-1.0, -1.0]  # Heavy penalty for rule violation
                self.done = True
                return (self._obs(0), self._obs(1)), rewards, self.done, False, {}

        # Check for invalid P1 decision attempt
        if active_player == 1 and self._is_decision_action(active_action, 1):
            rewards = [-1.0, -1.0]
            self.done = True
            return (self._obs(0), self._obs(1)), rewards, self.done, False, {}

        # Record the action in history
        if self.t < self.max_len:
            self.history[self.t, 0] = 1  # timestep active
            self.history[self.t, 1] = active_player  # whose turn
            self.history[self.t, 2] = active_action   # what action

        # Update last action for this player
        self.last_action[active_player] = active_action
        self.t += 1

        # Handle different action types
        if self._is_decision_action(active_action, active_player):
            # Player 0 makes decision
            guess = active_action - 2
            if guess in self.zero_idx:
                rewards = [1.0, 1.0]  # Success
            else:
                rewards = [-1.0, -1.0]  # Failure
            self.done = True

        elif self._is_pass_action(active_action, active_player):
            # Player passes turn to other player
            self.current_turn = 1 - self.current_turn
            # Small penalty for passing to encourage efficient communication
            rewards = [0.0, 0.0]

        else:
            # Player sends a bit (0 or 1)
            # Turn stays with same player - they can speak again!
            rewards = [0.05 * self.t, 0.05 * self.t]  # Remove the negative sign

        # Timeout penalty
        if not self.done and self.t >= self.max_len:
            self.done = True
            rewards = [-1.0, -1.0]

        self.total_steps += 1

        return (self._obs(0), self._obs(1)), rewards, self.done, False, {
            "current_turn": self.current_turn,
            "can_pass": {0: self._can_pass(0), 1: self._can_pass(1)}
        } 
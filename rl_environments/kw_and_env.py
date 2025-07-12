import numpy as np
import random
import gymnasium as gym
from typing import Optional

class KWAndEnv(gym.Env):
    """
    Two-player cooperative KW game for f(x)=AND(x).
    Player-0 holds x s.t. AND(x)=1  (i.e. x = 1…1)
    Player-1 holds y s.t. AND(y)=0  (i.e. y has ≥1 zero)
    Goal: find an index i with x_i=1, y_i=0.
    """

    def __init__(self, n: int = 4):
        super().__init__()
        self.n = n
        self.max_len = 2*n+1
        self.total_steps = 0
        self.hard_easy_prob = 0.5

        # Actions: {0: send '0', 1: send '1', 2…2+n-1: decide index i}
        # Player 0: can send bits (0,1) OR decide (2, 3, ..., n+1) or pass
        self.action_space_p0 = gym.spaces.Discrete(3 + n)
        # Player 1: can ONLY send bits (0,1) or pass - NO decision actions
        self.action_space_p1 = gym.spaces.Discrete(3)

        # Track player's turn
        self.current_turn = 0  # 0 = P0's turn, 1 = P1's turn

        # Last action of players (for "pass" tracking)
        self.last_action: dict[int, Optional[int]] = {0: None, 1: None}

        # Calculate observation space size
        history_size = self.max_len * 3  # 3 features per timestep
        obs_size = n + history_size  # n bits + history

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.done = False

        # Player-0 sees x = 1…1 ; Player-1 sees random y with ≥1 zero
        self.x = np.ones(self.n, dtype=np.int8)

        # Uniform
        # uniformly random subset of indices where x is 0
        # self.zero_idx = np.where(np.random.random(self.n) < 0.5)[0]

        # Random max term (AND is monotone)
        if np.random.random() < self.hard_easy_prob:
            self.zero_idx = np.random.choice(range(self.n), size=random.randint(1,1), replace=False)
        else:
            self.zero_idx = np.where(np.random.random(self.n) < 0.5)[0]

        # set the bits at the zero indices to 0 only on y
        self.y = np.ones(self.n, dtype=np.int8)
        self.y[self.zero_idx] = 0

        self.current_turn = 0
        self.last_action = {0: None, 1: None}

        # Initialize history properly
        self.history = np.zeros((self.max_len, 3), dtype=np.float32)

        return self._obs(0), self._obs(1)

    def _obs(self, player):
        """Return flattened observation"""
        player_input = self.x if player == 0 else self.y
        # Flatten history and concatenate with input
        flat_history = self.history.reshape(-1)  # This ensures we get the correct number of elements
        obs = np.concatenate([player_input.astype(np.float32), flat_history])
        return obs

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
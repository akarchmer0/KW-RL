import math
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class DQN(nn.Module):
    def __init__(self, input_size: int, action_size: int):
        super().__init__()
        # Dynamic hidden sizes based on observation size
        hidden_size1 = 256  # First hidden layer
        hidden_size2 = 128   # Second hidden layer
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, action_size)
        )

    def forward(self, x):
        return self.net(x)
    
class LSTMQNetwork(nn.Module):
    def __init__(self, input_size: int, action_size: int, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, action_size)
        )
        
    def forward(self, x, hidden=None):
        """
        x: (batch_size, seq_len, input_size)
        hidden: tuple of (h_0, c_0) each of shape (num_layers, batch_size, hidden_size)
        """
        lstm_out, hidden = self.lstm(x, hidden)
        # Use the last timestep output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        q_values = self.output_head(last_output)
        return q_values, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h_0, c_0)

class ReplayBuffer:
    def __init__(self, capacity=int(1e5)):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class KWAgent:
    def __init__(self, n, player_id, model_type='LSTM', lr=3e-4, ucb_c=2.0):
        self.n = n
        self.player_id = player_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Calculate observation size (same for both players)
        self.obs_size = n + (2 * n + 1) * 3
        
        # ASYMMETRIC ACTION SPACES
        if player_id == 0:
            # Player 0: bits (0,1) + decisions (2 to n+1) + pass (n+2)
            self.action_size = 3 + n  # 2 bits + n decisions + 1 pass
        else:
            # Player 1: bits (0,1) + pass (2) - NO decision actions
            self.action_size = 3      # 2 bits + 1 pass only
        
        # Initialize networks with player-specific action sizes
        if model_type == 'DQN':
            self.policy_net = DQN(self.obs_size, self.action_size).to(self.device)
            self.target_net = DQN(self.obs_size, self.action_size).to(self.device)
        elif model_type == 'LSTM':
            self.policy_net = LSTMQNetwork(self.obs_size, self.action_size).to(self.device)
            self.target_net = LSTMQNetwork(self.obs_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.memory = ReplayBuffer(capacity=int(5e4))
        
        # UCB parameters
        self.ucb_c = ucb_c  # Exploration parameter (typically 1-3)
        self.action_counts = np.zeros(self.action_size)  # Count of each action taken
        self.total_steps = 0  # Total number of actions taken
        
        # Initialize training parameters
        self.batch_size = 128
        self.gamma = 0.999  # Higher gamma like the original fast-learning code
    
    def select_action(self, state, training=True):
        """Select action using UCB (Upper Confidence Bound) policy."""
        if not training:
            # During evaluation, use pure exploitation (no exploration)
            with torch.no_grad():
                # Reshape state for LSTM (batch_size=1, sequence_length=1, features)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
                q_values, _ = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        
        # UCB action selection during training
        with torch.no_grad():
            # Reshape state for LSTM (batch_size=1, sequence_length=1, features)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            q_values, _ = self.policy_net(state_tensor)
            q_values = q_values.cpu().numpy().flatten()
        
        # Calculate UCB values for each action
        ucb_values = np.zeros(self.action_size)
        
        for action in range(self.action_size):
            if self.action_counts[action] == 0:
                # If action hasn't been tried, give it infinite confidence
                # This ensures all actions are tried at least once
                ucb_values[action] = float('inf')
            else:
                # UCB formula: Q(a) + c * sqrt(ln(t) / N(a))
                confidence_interval = self.ucb_c * math.sqrt(
                    math.log(self.total_steps + 1) / self.action_counts[action]
                )
                ucb_values[action] = q_values[action] + confidence_interval
        
        # Handle player-specific action preferences for untried actions
        if np.any(np.isinf(ucb_values)):
            # If some actions haven't been tried, prefer player-appropriate actions
            untried_actions = np.where(np.isinf(ucb_values))[0]
            if self.player_id == 1 and len(untried_actions) > 1:
                # Player 1 should prefer sending bits (actions 0, 1) over passing (action 2)
                bit_actions = [a for a in untried_actions if a in [0, 1]]
                if bit_actions:
                    selected_action = random.choice(bit_actions)
                else:
                    selected_action = random.choice(untried_actions)
            else:
                selected_action = random.choice(untried_actions)
        else:
            # Select action with highest UCB value
            selected_action = np.argmax(ucb_values)
        
        # Update counters
        self.action_counts[selected_action] += 1
        self.total_steps += 1
        
        return selected_action
    
    def store(self, state, action, reward, next_state, done):
        """Store transition in replay memory."""
        self.memory.push(state, action, reward, next_state, done)
    
    def optimize(self):
        """Perform one step of optimization."""
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state).unsqueeze(1).to(self.device)  # Add sequence dimension
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).unsqueeze(1).to(self.device)  # Add sequence dimension
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q_values, _ = self.policy_net(state_batch)
        current_q_values = current_q_values.gather(1, action_batch)
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            # Use policy network to select actions
            next_actions, _ = self.policy_net(next_state_batch)
            next_actions = next_actions.max(1)[1].unsqueeze(1)
            
            # Use target network to evaluate actions
            next_q_values, _ = self.target_net(next_state_batch)
            next_q_values = next_q_values.gather(1, next_actions)
            
            # Compute target Q values
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def update_target_network(self):
        """Update target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def reset_ucb_stats(self):
        """Reset UCB statistics (useful when starting a new training phase)."""
        self.action_counts = np.zeros(self.action_size)
        self.total_steps = 0
    
    def get_exploration_stats(self):
        """Get current exploration statistics."""
        return {
            'total_steps': self.total_steps,
            'action_counts': self.action_counts.copy(),
            'action_frequencies': self.action_counts / max(1, self.total_steps)
        }
    
    def save(self, path):
        """Save model."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n': self.n,
            'player_id': self.player_id,
            'ucb_c': self.ucb_c,
            'action_counts': self.action_counts,
            'total_steps': self.total_steps
        }, path)
    
    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.n = checkpoint['n']
        self.player_id = checkpoint['player_id']
        self.ucb_c = checkpoint.get('ucb_c', 2.0)
        self.action_counts = checkpoint.get('action_counts', np.zeros(self.action_size))
        self.total_steps = checkpoint.get('total_steps', 0)
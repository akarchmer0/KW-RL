#!/usr/bin/env python
# coding: utf-8

import os
import logging
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from rl_environments.kw_env import KWIPEnv, KWAndEnv, KWVPEnv, KWAndFourierEnv
from kw_agent import KWAgent

def setup_logging(output_dir):
    """Set up logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"kw_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This ensures output to console
        ]
    )
    
    return logging.getLogger(__name__)

def save_results(log_dir, returns, success_rates, n_bits, name):
    """Save training results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save returns
    returns_file = os.path.join(log_dir, f"returns_{timestamp}_{name}.npy")
    np.save(returns_file, np.array(returns))
    
    # Save success rates
    success_file = os.path.join(log_dir, f"success_rates_{timestamp}.npy")
    np.save(success_file, np.array(success_rates))
    
    # Save metadata
    metadata = {
        "n_bits": n_bits,
        "name": name,
        "timestamp": timestamp,
        "final_success_rate": success_rates[-1] if success_rates else None,
        "final_return": returns[-1] if returns else None,
        "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    }
    
    metadata_file = os.path.join(log_dir, f"metadata_{timestamp}.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

def transfer_weights(old_agent, new_agent):
    with torch.no_grad():
        # Get the old and new networks
        old_net = old_agent.policy_net
        new_net = new_agent.policy_net
        
        # Initialize new LSTM weights with Xavier
        for layer in range(old_net.lstm.num_layers):
            # Initialize input-to-hidden weights
            nn.init.xavier_uniform_(getattr(new_net.lstm, f'weight_ih_l{layer}'))
            nn.init.zeros_(getattr(new_net.lstm, f'bias_ih_l{layer}'))
            
            # Copy hidden-to-hidden weights (these don't depend on input size)
            setattr(new_net.lstm, f'weight_hh_l{layer}', getattr(old_net.lstm, f'weight_hh_l{layer}'))
            setattr(new_net.lstm, f'bias_hh_l{layer}', getattr(old_net.lstm, f'bias_hh_l{layer}'))
        
        # Initialize output head
        nn.init.xavier_uniform_(new_net.output_head[0].weight)  # First linear layer in output_head
        nn.init.zeros_(new_net.output_head[0].bias)
        nn.init.xavier_uniform_(new_net.output_head[3].weight)  # Second linear layer in output_head
        nn.init.zeros_(new_net.output_head[3].bias)
        
        # Copy target network
        new_agent.target_net.load_state_dict(new_agent.policy_net.state_dict())

def train_kw_agents(env, n_bits=4, num_episodes=1_000_000, output_dir="results",
                   eval_interval=10000, min_episodes=5000, prev_agents=None, model_type='LSTM'):
    """Train two agents to play the KW game.
    
    Args:
        env: The environment to train in
        n_bits: Number of bits in the game
        num_episodes: Maximum number of episodes to train for
        output_dir: Directory to save results
        eval_interval: How often to evaluate performance
        min_episodes: Minimum number of episodes before checking for 99% success
        prev_agents: Previously trained agents to use as initialization
    """
    # Initialize agents, either from scratch or from previous training
    if prev_agents is None:
        agents = [KWAgent(env, n_bits, player_id=i, model_type=model_type) for i in range(2)]
    else:
        agents = []
        for i, prev_agent in enumerate(prev_agents):
            # Create new agent with larger n
            new_agent = KWAgent(env, n_bits, player_id=i, model_type=model_type)
            # Transfer weights from previous agent to new agent
            transfer_weights(prev_agent, new_agent)
            agents.append(new_agent)

            
    # Debug logging for dimensions
    logger.info(f"\nEnvironment observation space shape: {env.observation_space.shape}")
    logger.info(f"Agent 0 observation size: {agents[0].obs_size}")
    logger.info(f"Agent 1 observation size: {agents[1].obs_size}")
    logger.info(f"Agent 0 action size: {agents[0].action_size}")
    logger.info(f"Agent 1 action size: {agents[1].action_size}")

    episode_returns = []
    success_rate_history = []
    epsilion_history = []
    
    # Variables for tracking high performance
    high_performance_episode = None
    episodes_after_high_performance = 0
    target_success_rate = 0.99
    episodes_to_continue = 100

    # Log device information
    device = agents[0].device
    logger.info(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    logger.info(f"\nStarting training for n={n_bits}")
    logger.info(f"Number of episodes: {num_episodes}")
    logger.info(f"Will continue for {episodes_to_continue} episodes after reaching {target_success_rate:.1%} success rate")
    logger.info(f"Evaluating every {eval_interval} episodes")
    logger.info("\n" + "="*50)

    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        episode_return = 0
        done = False

        while not done:
            actions = [agents[i].select_action(obs[i]) for i in range(2)]
            next_obs, rewards, done, _, info = env.step(tuple(actions))

            for i in range(2):
                agents[i].store(obs[i], actions[i], rewards[i], next_obs[i], done)
                agents[i].optimize()

            obs = next_obs
            episode_return += rewards[0]

        episode_returns.append(episode_return)
        # epsilion_history.append(agents[0].epsilon_current) # epsilon-greedy only

        # Update target networks periodically
        if episode % 500 == 0:
            for agent in agents:
                agent.update_target_network()

        # Evaluate and log periodically
        if (episode + 1) % eval_interval == 0:
            success_rate = evaluate_agents(env, agents, episodes=2_000)
            success_rate_history.append(success_rate)
            
            # Calculate average return over recent episodes
            recent_returns = episode_returns[-eval_interval:]
            avg_return = sum(recent_returns) / len(recent_returns)
            
            # Check for high performance after minimum episodes
            if episode >= min_episodes:
                if success_rate >= target_success_rate:
                    if high_performance_episode is None:
                        high_performance_episode = episode
                        logger.info(f"\nReached {target_success_rate:.1%} success rate at episode {episode+1}")
                    episodes_after_high_performance = episode - high_performance_episode
                elif high_performance_episode is not None:
                    episodes_after_high_performance = episode - high_performance_episode
                
                if high_performance_episode is not None:
                    logger.info(f"Episodes since reaching {target_success_rate:.1%}: {episodes_after_high_performance}")
                    if episodes_after_high_performance >= episodes_to_continue:
                        logger.info(f"\nCompleted {episodes_to_continue} episodes after reaching {target_success_rate:.1%} success rate")
                        break
            
            # Print progress with clear formatting
            logger.info(f"\nEpisode {episode+1}/{num_episodes}")
            logger.info(f"Success Rate: {success_rate:.1%}")
            logger.info(f"Average Return: {avg_return:.3f}")
            # logger.info(f"Epsilon: {agents[0].epsilon_current:.3f}") # epsilon-greedy only
            logger.info("-"*30)

    return agents, episode_returns, success_rate_history

def evaluate_agents(env, agents, episodes=1000, verbose=False, show_transcripts=10):
    successes = 0
    transcripts_shown = 0
    episode_lengths = []
    transcript_lengths = []
    pass_counts = []
    decision_outcomes = {"correct": 0, "wrong": 0, "timeout": 0, "invalid_p1": 0}

    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_actions = []

        show_this_episode = verbose and transcripts_shown < show_transcripts

        if show_this_episode:
            logger.info(f"\n=== Episode {episode + 1} Transcript ===")
            logger.info(f"Player 0 sees x: {env.x}")
            logger.info(f"Player 1 sees y: {env.y}")

        step_num = 0
        bits_num = 0
        passes_num = 0
        current_turn = 0

        while not done:
            actions = [agents[i].select_action(obs[i], training=False) for i in range(2)]
            episode_actions.append(actions)
            next_obs, rewards, done, _, info = env.step(tuple(actions))

            step_num += 1

            # Analyze active player's action
            active_action = actions[current_turn]

            if show_this_episode:
                if env._is_pass_action(active_action, current_turn):
                    action_desc = "PASS turn"
                    passes_num += 1
                elif env._is_decision_action(active_action, current_turn):
                    action_desc = f"DECIDE index {active_action - 2}"
                else:
                    action_desc = f"send bit {active_action}"
                    bits_num += 1

                logger.info(f"  Turn {step_num}: Player {current_turn} -> {action_desc}")

                if done:
                    if env._is_decision_action(active_action, current_turn):
                        guess = active_action - 2
                        correct = rewards[0] > 0
                        logger.info(f"  → P0 guessed index {guess}: {'CORRECT!' if correct else 'WRONG!'}")
                        if correct:
                            decision_outcomes["correct"] += 1
                        else:
                            decision_outcomes["wrong"] += 1
                    elif step_num >= env.max_len:
                        logger.info(f"  → TIMEOUT after {env.max_len} turns")
                        decision_outcomes["timeout"] += 1
            else:
                # Count for non-verbose episodes
                if env._is_pass_action(active_action, current_turn):
                    passes_num += 1
                elif not env._is_decision_action(active_action, current_turn):
                    bits_num += 1

            # Update current turn for next iteration
            if not done and "current_turn" in info:
                current_turn = info["current_turn"]

            obs = next_obs

        episode_lengths.append(step_num)
        transcript_lengths.append(bits_num)
        pass_counts.append(passes_num)

        if rewards[0] > 0:
            successes += 1

        if show_this_episode:
            result = "SUCCESS" if rewards[0] > 0 else "FAILURE"
            logger.info(f"Result: {result} (turns: {step_num}, bits: {bits_num}, passes: {passes_num})")
            transcripts_shown += 1

    success_rate = successes / episodes

    if verbose:
        logger.info(f"\n=== Evaluation Summary ===")
        logger.info(f"Success rate: {success_rate:.1%} ({successes}/{episodes})")
        logger.info(f"Average episode length: {sum(episode_lengths)/len(episode_lengths):.1f} turns")
        logger.info(f"Average bits communicated: {sum(transcript_lengths)/len(transcript_lengths):.1f} bits")
        logger.info(f"Average passes per episode: {sum(pass_counts)/len(pass_counts):.1f} passes")

        logger.info(f"\nCommunication Analysis:")
        avg_bits_per_turn = sum(transcript_lengths) / sum(episode_lengths) if sum(episode_lengths) > 0 else 0
        logger.info(f"  Average bits per turn: {avg_bits_per_turn:.1f}")

    return success_rate

if __name__ == "__main__":
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Values of n to try
    n_values = [4, 8]
    envs = ['AND']
    env = None
    num_episodes = 100_000

    name = f"n{n_values[0]}_env{envs[0]}_episodes{num_episodes}"

    for env_name in envs:   
        # Keep track of previously trained agents for curriculum learning
        prev_agents = None
        
        for n in n_values:
            if env_name == 'IP':
                env = KWIPEnv(n=n)
            elif env_name == 'AND':
                env = KWAndEnv(n=n)
            elif env_name == 'AND_FOURIER':
                env = KWAndFourierEnv(n=n)
            elif env_name == 'VP':
                env = KWVPEnv(n=n)
            else:
                raise ValueError(f"Invalid environment: {env}")

            print(f"\n{'='*50}")
            print(f"Starting experiment for n={n}, env={env_name}")
            if prev_agents is not None:
                print("Using previously trained agents as initialization")
            print(f"{'='*50}\n")
            
            # Setup logging for this n value
            logger = setup_logging(output_dir)

            # Train agents, passing previous agents for curriculum learning
            trained_agents, returns, success_rates = train_kw_agents(
                env=env,
                n_bits=n,
                num_episodes=num_episodes,
                output_dir=output_dir,
                eval_interval=5000, 
                min_episodes=10000,
                prev_agents=prev_agents
            )
            
            # Save results
            save_results(output_dir, returns, success_rates, n, name)

            # Final evaluation
            final_success_rate = evaluate_agents(env, trained_agents, episodes=2000, verbose=True, show_transcripts=10)
            print(f"\nFinal Success Rate for n={n}: {final_success_rate:.1%}")

            # Save the trained models
            for i, agent in enumerate(trained_agents):
                model_path = os.path.join(output_dir, f"agent_{i}_n{n}.pt")
                agent.save(model_path)
                print(f"Saved model for agent {i} to {model_path}")
            
            # Store these agents for the next iteration
            prev_agents = trained_agents






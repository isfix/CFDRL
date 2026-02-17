# src/trainer.py
"""
DDQN Trainer with TradingEnvironment.

Architecture:
- TradingEnvironment: sequential env with position tracking
- Semantic actions: HOLD(0), BUY(1), SELL(2), CLOSE(3)
- Position-aware QNetwork (trade_state in observation)
- Cost on position change only, per-step epsilon decay, gamma=0.95
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import MetaTrader5 as mt5
from tqdm import tqdm
import argparse
from config import Settings
from src.data_factory import fetch_data, prepare_features, FEATURES, resample_ohlcv
from src.brain import QNetwork, TRADE_STATE_DIM
from src.per_memory import PrioritizedReplayBuffer
from src.utils import logger
import random

# Action constants
ACT_HOLD = 0
ACT_BUY = 1
ACT_SELL = 2
ACT_CLOSE = 3


class TradingEnvironment:
    """
    Sequential trading environment with position tracking.
    
    State = (market_features_sequence, trade_state)
    Trade state = [current_position, bars_held_normalized, unrealized_pnl_normalized]
    
    Actions: HOLD(0), BUY(1), SELL(2), CLOSE(3)
    - HOLD: maintain current position
    - BUY: go long (or do nothing if already long)
    - SELL: go short (or do nothing if already short)
    - CLOSE: close current position (or do nothing if flat)
    
    Reward:
    - Per-bar unrealized PnL when holding a position
    - Transaction cost only on position changes (entry/exit/reversal)
    """
    
    def __init__(self, features, close_prices, seq_len, cost_bps=2, scaling_factor=10000.0):
        self.features = features        # (T, num_features) numpy array
        self.close_prices = close_prices  # (T,) numpy array
        self.seq_len = seq_len
        self.cost_fraction = cost_bps * 0.0001
        self.scaling_factor = scaling_factor
        
        # Episode state
        self.position = 0.0     # -1 (short), 0 (flat), +1 (long)
        self.entry_price = 0.0
        self.bars_held = 0
        self.step_idx = seq_len  # Start after warmup
        self.done = False
    
    def reset(self, start_idx=None):
        """Reset environment to beginning of episode."""
        self.position = 0.0
        self.entry_price = 0.0
        self.bars_held = 0
        self.step_idx = start_idx if start_idx else self.seq_len
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        """Return current state: (features_seq, trade_state)."""
        # Market features: last seq_len bars
        start = self.step_idx - self.seq_len
        end = self.step_idx
        market_seq = self.features[start:end]  # (seq_len, num_features)
        
        # Trade state: position info
        current_price = self.close_prices[self.step_idx - 1]
        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl = self.position * (current_price - self.entry_price) * self.scaling_factor
        else:
            unrealized_pnl = 0.0
        
        trade_state = np.array([
            self.position,                         # -1, 0, +1
            min(self.bars_held / 48.0, 1.0),       # Normalized: 0-48 bars -> 0-1 (12 hours)
            np.clip(unrealized_pnl / 100.0, -1, 1) # Normalized unrealized PnL
        ], dtype=np.float32)
        
        return market_seq, trade_state
    
    def step(self, action):
        """
        Execute action, return (next_state, reward, done, info).
        
        Reward logic:
        - Per-bar: position * price_change * scaling_factor
        - Cost: charged only when position changes
        """
        if self.done:
            return self._get_state(), 0.0, True, {}
        
        current_price = self.close_prices[self.step_idx - 1]
        old_position = self.position
        
        # Execute action
        new_position = self._execute_action(action, current_price)
        position_changed = (new_position != old_position)
        
        # Advance to next bar
        self.step_idx += 1
        if self.step_idx >= len(self.close_prices):
            self.done = True
            # Force close on episode end
            if self.position != 0:
                reward = self._close_position(self.close_prices[-1])
            else:
                reward = 0.0
            return self._get_state() if not self.done else (self.features[-self.seq_len:], np.zeros(3, dtype=np.float32)), reward, True, {'forced_close': True}
        
        next_price = self.close_prices[self.step_idx - 1]
        
        # Calculate reward
        reward = 0.0
        
        # Transaction cost on position change
        if position_changed:
            cost = self.cost_fraction * self.scaling_factor * abs(new_position - old_position)
            reward -= cost
        
        # Per-bar PnL from holding position
        if self.position != 0:
            price_change = (next_price - current_price) * self.scaling_factor
            reward += self.position * price_change
            self.bars_held += 1
        
        # Clip reward
        reward = np.clip(reward, -Settings.REWARD_CLIP, Settings.REWARD_CLIP)
        
        next_state = self._get_state()
        info = {
            'position': self.position,
            'position_changed': position_changed,
            'price': next_price,
        }
        
        return next_state, reward, self.done, info
    
    def _execute_action(self, action, current_price):
        """Apply semantic action to position."""
        if action == ACT_HOLD:
            pass  # Do nothing
        elif action == ACT_BUY:
            if self.position <= 0:
                # Close short if any, then go long
                self.position = 1.0
                self.entry_price = current_price
                self.bars_held = 0
        elif action == ACT_SELL:
            if self.position >= 0:
                # Close long if any, then go short
                self.position = -1.0
                self.entry_price = current_price
                self.bars_held = 0
        elif action == ACT_CLOSE:
            if self.position != 0:
                self.position = 0.0
                self.entry_price = 0.0
                self.bars_held = 0
        
        return self.position
    
    def _close_position(self, close_price):
        """Close position and return final PnL."""
        if self.position == 0:
            return 0.0
        pnl = self.position * (close_price - self.entry_price) * self.scaling_factor
        cost = self.cost_fraction * self.scaling_factor
        self.position = 0.0
        self.entry_price = 0.0
        self.bars_held = 0
        return np.clip(pnl - cost, -Settings.REWARD_CLIP, Settings.REWARD_CLIP)


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on device: {self.device}")
        self.loss_fn = nn.SmoothL1Loss(reduction='none')  # Huber loss for stability

    def soft_update(self, policy_net, target_net, tau=Settings.TAU):
        """Soft-update target network: theta_target = tau*theta_policy + (1-tau)*theta_target"""
        for tp, tt in zip(policy_net.parameters(), target_net.parameters()):
            tt.data.copy_(tau * tp.data + (1.0 - tau) * tt.data)

    def train_model(self, symbol: str):
        """Main training loop with sequential environment."""
        logger.info(f"Phase B Training: {symbol}")
        
        pair_cfg = Settings.PAIR_CONFIGS.get(symbol, Settings.PAIR_CONFIGS['EURUSD'])
        SCALING_FACTOR = pair_cfg['scaling_factor']
        
        # ---- Fetch and prepare data ----
        logger.info("Fetching data from MT5...")
        raw_df = fetch_data(symbol, Settings.TRAIN_DATA_BARS)
        if raw_df.empty:
            logger.error("No data fetched. Exiting.")
            return
        
        # Resample M5 -> M15
        logger.info("Resampling to M15...")
        df = resample_ohlcv(raw_df, '15min')
        logger.info(f"M15 bars: {len(df)}")
        
        # Compute features
        logger.info("Computing features (35 features)...")
        df = prepare_features(df)
        logger.info(f"Bars after features: {len(df)}")
        
        # Extract arrays
        available_feats = [f for f in FEATURES if f in df.columns]
        if len(available_feats) < 30:
            logger.warning(f"Only {len(available_feats)}/35 features available")
        
        feature_data = df[available_feats].values.astype(np.float32)
        close_prices = df['close'].values.astype(np.float64)
        
        # ---- Train/Val Split ----
        total = len(feature_data)
        train_end = int(total * 0.8)
        val_end = total
        
        logger.info(f"Train: 0-{train_end} ({train_end} bars)")
        logger.info(f"Val: {train_end}-{val_end} ({val_end - train_end} bars)")
        
        # ---- Create Networks ----
        input_dim = len(available_feats)
        policy_net = QNetwork(
            input_dim=input_dim,
            output_dim=Settings.OUTPUT_DIM,
            trade_state_dim=TRADE_STATE_DIM
        ).to(self.device)
        target_net = copy.deepcopy(policy_net).to(self.device)
        target_net.eval()
        
        optimizer = optim.Adam(policy_net.parameters(), lr=Settings.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=Settings.LR_SCHEDULER_PATIENCE,
            factor=Settings.LR_SCHEDULER_FACTOR
        )
        
        # ---- PER Buffer ----
        memory = PrioritizedReplayBuffer(
            capacity=Settings.MEMORY_CAPACITY,
            alpha=Settings.PER_ALPHA
        )
        
        # ---- Training Loop ----
        epsilon = Settings.EPSILON_START
        total_steps = 0
        best_val_sharpe = -np.inf
        patience_counter = 0
        
        for epoch in range(1, Settings.EPOCHS + 1):
            policy_net.train()
            
            # Create training environment
            env = TradingEnvironment(
                features=feature_data[:train_end],
                close_prices=close_prices[:train_end],
                seq_len=Settings.SEQUENCE_LENGTH,
                cost_bps=Settings.TRANSACTION_COST_BPS,
                scaling_factor=SCALING_FACTOR
            )
            
            market_seq, trade_state = env.reset()
            episode_reward = 0.0
            episode_trades = 0
            episode_steps = 0
            losses = []
            
            # Progress bar
            max_steps = train_end - Settings.SEQUENCE_LENGTH - 1
            pbar = tqdm(range(max_steps), desc=f"Epoch {epoch}/{Settings.EPOCHS}", ncols=100)
            
            for step in pbar:
                # Convert state to tensors
                market_tensor = torch.FloatTensor(market_seq).unsqueeze(0).to(self.device)
                trade_tensor = torch.FloatTensor(trade_state).unsqueeze(0).to(self.device)
                
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = random.randint(0, Settings.OUTPUT_DIM - 1)
                else:
                    with torch.no_grad():
                        q_values = policy_net(market_tensor, trade_tensor)
                        action = q_values.argmax(dim=1).item()
                
                # Execute action
                (next_market_seq, next_trade_state), reward, done, info = env.step(action)
                
                if info.get('position_changed', False):
                    episode_trades += 1
                
                # Store in PER buffer: pack (market_seq, trade_state) as composite state
                state = (market_seq.copy(), trade_state.copy())
                next_state_packed = (next_market_seq.copy(), next_trade_state.copy())
                memory.push(state, action, reward, next_state_packed, done)
                
                # Update state
                market_seq = next_market_seq
                trade_state = next_trade_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # Per-step epsilon decay (Phase B FIX #3)
                if Settings.EPSILON_DECAY_PER_STEP:
                    epsilon = max(Settings.EPSILON_MIN, epsilon * Settings.EPSILON_DECAY)
                
                # ---- Train from replay buffer ----
                if len(memory) >= Settings.BATCH_SIZE and total_steps % 4 == 0:
                    loss = self._train_step(policy_net, target_net, optimizer, memory)
                    losses.append(loss)
                
                # Soft-update target network
                if total_steps % Settings.TARGET_UPDATE_FREQ == 0:
                    self.soft_update(policy_net, target_net)
                
                # Update progress bar
                if step % 100 == 0:
                    avg_loss = np.mean(losses[-100:]) if losses else 0
                    pbar.set_postfix({
                        'R': f'{episode_reward:.1f}',
                        'T': episode_trades,
                        'eps': f'{epsilon:.3f}',
                        'L': f'{avg_loss:.4f}'
                    })
                
                if done:
                    break
            
            pbar.close()
            
            # ---- Epoch Summary ----
            avg_loss = np.mean(losses) if losses else 0
            logger.info(f"Epoch {epoch}: reward={episode_reward:.1f}, trades={episode_trades}, "
                       f"epsilon={epsilon:.4f}, avg_loss={avg_loss:.4f}, steps={episode_steps}")
            
            # ---- Validation ----
            val_sharpe, val_return = self.validate(
                policy_net, feature_data[train_end:val_end],
                close_prices[train_end:val_end],
                SCALING_FACTOR
            )
            logger.info(f"  Val Sharpe: {val_sharpe:.3f}, Val Return: {val_return:.2f}")
            
            scheduler.step(val_sharpe)
            
            # ---- Checkpointing ----
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                patience_counter = 0
                save_path = os.path.join("models", f"best_{symbol}_phaseB.pt")
                os.makedirs("models", exist_ok=True)
                torch.save(policy_net.state_dict(), save_path)
                logger.info(f"  New best! Saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= Settings.EARLY_STOP_PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        logger.info(f"Training complete. Best val Sharpe: {best_val_sharpe:.3f}")
    
    def _train_step(self, policy_net, target_net, optimizer, memory):
        """One training step from replay buffer."""
        states, actions, rewards, next_states, dones, indices, weights = memory.sample(
            Settings.BATCH_SIZE, beta=Settings.PER_BETA_START
        )
        
        # Unpack composite states: each state is (market_seq, trade_state)
        market_seqs = np.array([s[0] for s in states])
        trade_states_arr = np.array([s[1] for s in states])
        next_market_seqs = np.array([s[0] for s in next_states])
        next_trade_states_arr = np.array([s[1] for s in next_states])
        
        market_t = torch.FloatTensor(market_seqs).to(self.device)
        trade_t = torch.FloatTensor(trade_states_arr).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_market_t = torch.FloatTensor(next_market_seqs).to(self.device)
        next_trade_t = torch.FloatTensor(next_trade_states_arr).to(self.device)
        dones_t = torch.FloatTensor(dones.astype(np.float32)).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        q_values = policy_net(market_t, trade_t)
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # DDQN target: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_q_policy = policy_net(next_market_t, next_trade_t)
            best_actions = next_q_policy.argmax(dim=1, keepdim=True)
            next_q_target = target_net(next_market_t, next_trade_t)
            next_q = next_q_target.gather(1, best_actions).squeeze(1)
            target = rewards_t + Settings.GAMMA * next_q * (1.0 - dones_t)
        
        # Huber loss with PER weights
        td_errors = q_selected - target
        loss = (self.loss_fn(q_selected, target) * weights_t).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()
        
        # Update priorities
        memory.update_priorities(indices, td_errors.abs().detach().cpu().numpy() + 1e-6)
        
        return loss.item()

    def validate(self, policy_net, features, close_prices, scaling_factor):
        """Run validation episode and compute Sharpe ratio."""
        policy_net.eval()
        
        env = TradingEnvironment(
            features=features,
            close_prices=close_prices,
            seq_len=Settings.SEQUENCE_LENGTH,
            cost_bps=Settings.TRANSACTION_COST_BPS,
            scaling_factor=scaling_factor
        )
        
        market_seq, trade_state = env.reset()
        daily_returns = []
        bar_return = 0.0
        bar_count = 0
        bars_per_day = 96  # 24h * 4 bars/hour for M15
        
        with torch.no_grad():
            while not env.done:
                market_tensor = torch.FloatTensor(market_seq).unsqueeze(0).to(self.device)
                trade_tensor = torch.FloatTensor(trade_state).unsqueeze(0).to(self.device)
                
                q_values = policy_net(market_tensor, trade_tensor)
                action = q_values.argmax(dim=1).item()
                
                (market_seq, trade_state), reward, done, info = env.step(action)
                bar_return += reward
                bar_count += 1
                
                # Aggregate to daily returns
                if bar_count >= bars_per_day:
                    daily_returns.append(bar_return)
                    bar_return = 0.0
                    bar_count = 0
                
                if done:
                    if bar_count > 0:
                        daily_returns.append(bar_return)
                    break
        
        policy_net.train()
        
        if len(daily_returns) < 5:
            return 0.0, 0.0
        
        daily_returns = np.array(daily_returns)
        total_return = daily_returns.sum()
        
        # Annualized Sharpe
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        return sharpe, total_return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase B Trainer")
    parser.add_argument("--pair", type=str, default="EURUSD", help="Symbol to train")
    args = parser.parse_args()
    symbol = args.pair

    if not mt5.initialize(path=Settings.MT5_PATH, login=Settings.MT5_LOGIN,
                         password=Settings.MT5_PASSWORD, server=Settings.MT5_SERVER):
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        exit()

    trainer = Trainer()
    
    try:
        trainer.train_model(symbol)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        mt5.shutdown()

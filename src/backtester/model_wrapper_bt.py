import numpy as np
import torch
import os
import pickle
import io
import traceback
from typing import Optional

class ModelWrapper:
    def __init__(self, artifacts_path: str = 'model_files/', logger=None):
        self.artifacts_path = artifacts_path
        self.log = logger if logger is not None else print
        self.model = None
        self.device = torch.device('cpu')
        self.hidden_states = None
        self.loaded = False
        self.obs_mean = None
        self.obs_var = None
        self.prediction_count = 0
        self.LoadModel()

    def LoadModel(self):
        try:
            with open(os.path.join(self.artifacts_path, "policy_weights.pth"), 'rb') as f:
                policy_weights = torch.load(f, map_location=self.device)

            class SafeUnpickler(pickle.Unpickler):
                class _Dummy:
                    def __init__(self, *args, **kwargs): pass
                    def __setstate__(self, state): pass
                def find_class(self, module, name):
                    if module.startswith("numpy.random"):
                        return self._Dummy
                    return super().find_class(module, name)

            with open(os.path.join(self.artifacts_path, "normalization_stats.pkl"), 'rb') as f:
                norm_stats = SafeUnpickler(f).load()

            with open(os.path.join(self.artifacts_path, "architecture_info.pkl"), 'rb') as f:
                arch_info = SafeUnpickler(f).load()

            obs_dim = arch_info.get('observation_dim',
                                     arch_info.get('obs_space')[0] if 'obs_space' in arch_info else None)
            action_dim = arch_info.get('action_dim',
                                       arch_info.get('act_space')[0] if 'act_space' in arch_info else None)
            hidden_dim = arch_info.get('hidden_dim',
                                       arch_info.get('lstm_hidden') if 'lstm_hidden' in arch_info else None)
            if None in (obs_dim, action_dim, hidden_dim):
                raise KeyError("architecture_info.pkl is missing mandatory dimension fields.")

            self.model = RecurrentPPOModel(obs_dim=obs_dim,
                                           action_dim=action_dim,
                                           lstm_hidden_size=hidden_dim)

            missing, unexpected = self.model.load_state_dict(policy_weights, strict=False)
            print(f'Actor weights loaded; missing={missing}, unexpected={unexpected}')
            self.model.to(self.device)

            if isinstance(norm_stats, dict):
                self.obs_mean = norm_stats['obs_mean']
                self.obs_var  = norm_stats['obs_var']
            elif hasattr(norm_stats, "obs_rms"):
                self.obs_mean = norm_stats.obs_rms.mean
                self.obs_var  = norm_stats.obs_rms.var
            else:
                raise TypeError(f"Unrecognised normalization object of type {type(norm_stats)}")

            self.obs_mean = self.obs_mean.reshape(-1).astype(np.float32)
            self.obs_var  = self.obs_var.reshape(-1).astype(np.float32)

            self.model.eval()
            self.reset_hidden_states()
            self.loaded = True
            print(f"Model loaded successfully. Obs dim: {obs_dim}, "
                     f"Action dim: {action_dim}, Hidden dim: {hidden_dim}")

        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            self.loaded = False

    def predict(self, observation: np.ndarray) -> Optional[np.ndarray]:
        if not self.loaded or self.model is None:
            self.log("Model not loaded, cannot make prediction")
            return None
        try:
            if self.obs_mean is None or self.obs_var is None:
                self.log("ERROR: Normalization statistics not loaded!")
                return None

            normalized_obs = (observation - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
            obs_tensor     = torch.FloatTensor(normalized_obs).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if self.hidden_states is None:
                    self.reset_hidden_states()
                action, self.hidden_states = self.model(obs_tensor, self.hidden_states)
                action = action.cpu().view(-1).numpy()

            return np.clip(action, -1, 1)

        except Exception as e:
            self.log(f"Error in model prediction: {str(e)}")
            return None

    def reset_hidden_states(self):
        if self.loaded and self.model is not None:
            hidden_dim = self.model.lstm_actor.hidden_size
            h_0 = torch.zeros(1, 1, hidden_dim).to(self.device)
            c_0 = torch.zeros(1, 1, hidden_dim).to(self.device)
            self.hidden_states = (h_0, c_0)

class RecurrentPPOModel(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, lstm_hidden_size: int):
        super().__init__()
        self.lstm_actor = torch.nn.LSTM(obs_dim, lstm_hidden_size, batch_first=True)
        self.mlp_extractor_policy_net = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU()
        )
        self.action_net = torch.nn.Linear(64, action_dim)
        self.log_std    = torch.nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs, hidden_states):
        lstm_out, new_hidden_states = self.lstm_actor(obs, hidden_states)
        features_in   = lstm_out[:, -1, :]
        features_out  = self.mlp_extractor_policy_net(features_in)
        action_means  = self.action_net(features_out)
        actions       = torch.tanh(action_means)
        return actions, new_hidden_states

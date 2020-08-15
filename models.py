from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
  x_sizes = tuple(map(lambda x: x.size(), x_tuple))
  y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
  y_size = y.size()
  return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])


class TransitionModel(jit.ScriptModule):
  __constants__ = ['min_std_dev']

  def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
    super().__init__()
    self.act_fn                    = getattr(F, activation_function)
    self.min_std_dev               = min_std_dev
    self.fc_embed_state_action     = nn.Linear(state_size + action_size, belief_size)
    self.rnn                       = nn.GRUCell(belief_size, belief_size)
    self.fc_embed_belief_prior     = nn.Linear(belief_size, hidden_size)
    self.fc_state_prior            = nn.Linear(hidden_size, 2 * state_size)
    self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
    self.fc_state_posterior        = nn.Linear(hidden_size, 2 * state_size)

  # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
  # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
  # t :  0  1  2  3  4  5
  # o :    -X--X--X--X--X-
  # a : -X--X--X--X--X-
  # n : -X--X--X--X--X-
  # pb: -X-
  # ps: -X-
  # b : -x--X--X--X--X--X-
  # s : -x--X--X--X--X--X-
  @jit.script_method
  def forward(self, prev_state:torch.Tensor, actions:torch.Tensor, prev_belief:torch.Tensor, observations:Optional[torch.Tensor]=None, nonterminals:Optional[torch.Tensor]=None) -> List[torch.Tensor]:
    # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
    # ----prev_state               [ s_{t-1} ]:   [batch_size, belief_size]
    # ----action           [ a_{t-1 : t+L-1} ]:   [chunk_size-1, batch_size, action_size]
    # ----prev_belief                [h_{t-1}]:   [batch_size, hidden_size]
    # ----encoded observations  [o_{t : t+L} ]:   [chunk_size-1, batch_size, embedding_size]


    T = actions.size(0) + 1
    beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
    beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state

    # Loop over time sequence
    for t in range(T - 1):
      _state = prior_states[t] if observations is None else posterior_states[t]  # Select appropriate previous state
      _state = _state if nonterminals is None else _state * nonterminals[t]  # Mask if previous transition was terminal

      # COMPUTE BELIEF [h_t] (deterministic hidden state)
      #   h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
      #---------------------------------------------------------
      # h_{t-1} <--  ACT FUNCTION <-- EMBEDDING  <-- concat(s_{t-1}, a_{t-1})
      # beliefs[t+1] == h_t <-- RNN( h_{t-1} )

      embeded_s      = self.fc_embed_state_action(torch.cat([_state, actions[t]],dim=1))
      hidden         = self.act_fn(embeded_s)
      beliefs[t + 1] = self.rnn(hidden, beliefs[t])


      # COMPUTE STATE PRIOR [s_t] by applying transition dynamics
      #  s_t ~ p(s_t | h_t )
      #--------------------------------------------------------
      #          s_t  <-- mu_s_t + sigma_s_t*N(0,1)  <-- mu_s_t <-- EMBEDDING <-- activation function <-- EMBEDDING <-- h_t
      #                          ^                                      |
      #                         /                                       |
      #            ____________/                                        |
      #           /                                                     |
      #          /                                                      v
      #   sigma_s_t  <-- softplus(sigma_s_t_prime) + min_sigma <-- sigma_s_t_prime

      hidden                             = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
      prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
      prior_std_devs[t + 1]              = F.softplus(_prior_std_dev) + self.min_std_dev
      prior_states[t + 1]                = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])

      if observations is not None:
        # COMPUTE STATE POSTERIOR [~o_t] by applying transition dynamics and using current observation
        #  post_s_t ~ p(s_t | h_t,o_{t-1} )
        #--------------------------------------------------------
        #  post_s_t  <-- mu_post_s_t + sigma_post_s_t*N(0,1)  <-- mu_post_s_t <-- EMBEDDING <-- activation function <-- EMBEDDING <-- concat( h_t, o_{t-1} )
        #                                    ^                                      |
        #                                   /                                       |
        #                      ____________/                                        |
        #                     /                                                     |
        #                    /                                                      v
        #       sigma_post_s_t  <-- softplus(sigma_post_s_t_prime) + min_sigma <-- sigma_post_s_t_prime

        t_ = t - 1  # Use t_ to deal with different time indexing for observations
        hidden                                     = self.act_fn(self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
        posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
        posterior_std_devs[t + 1]                  = F.softplus(_posterior_std_dev) + self.min_std_dev
        posterior_states[t + 1]                    = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])


    # RETURN NEW HIDDEN STATES
    # [  h_{t : t+L}, s_{t : t+L}, mu_s_{t : t+L}, sigma_s_{t : t+L}]
    # If observations:
    #        # [  h_{t : t+L}, s_{t : t+L}, mu_s_{t : t+L}, sigma_s_{t : t+L}, post_s_{ t : t+L }, mu_post_s__{ t : t+L }, sigma_post_s_{ t : t+L }]
    #--------------------------------------------------------------

    hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
    if observations is not None:
      hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]

    return hidden


class SymbolicObservationModel(jit.ScriptModule):
  def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, observation_size)

  @jit.script_method
  def forward(self, belief, state):
    hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
    hidden = self.act_fn(self.fc2(hidden))
    observation = self.fc3(hidden)
    return observation


class VisualObservationModel(jit.ScriptModule):
  __constants__ = ['embedding_size']
  
  def __init__(self, belief_size, state_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.fc1   = nn.Linear(belief_size + state_size, embedding_size)
    self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
    self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
    self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
    self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

  @jit.script_method
  def forward(self, belief, state):
    # ~o_t ~ p(o_t | h_t, post_s_t)
    # ---------------------------
    #  ~o_t <-- CONVOLUTIONS AND ACT FUNCTIONS <-- EMBEDDING <-- concat(h_t,s_t)


    hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
    hidden = hidden.view(-1, self.embedding_size, 1, 1)
    hidden = self.act_fn(self.conv1(hidden))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    observation = self.conv4(hidden)

    return observation


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
  if symbolic:
    return SymbolicObservationModel(observation_size, belief_size, state_size, embedding_size, activation_function)
  else:
    return VisualObservationModel(belief_size, state_size, embedding_size, activation_function)


class RewardModel(jit.ScriptModule):
  def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)

  @jit.script_method
  def forward(self, belief, state):
    # ~r_t ~ p(r_t | h_t, post_s_t)
    # ---------------------------
    #  ~r_t <-- LINEAR <-- ACT FUNCTION <-- LINEAR <-- ACT_FUNCTION <-- LINEAR <-- concat(h_t, post_s_t)

    hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
    hidden = self.act_fn(self.fc2(hidden))
    reward = self.fc3(hidden).squeeze(dim=1)
    return reward


class SymbolicEncoder(jit.ScriptModule):
  def __init__(self, observation_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(observation_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, embedding_size)

  @jit.script_method
  def forward(self, observation):
    hidden = self.act_fn(self.fc1(observation))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.fc3(hidden)
    return hidden


class VisualEncoder(jit.ScriptModule):
  __constants__ = ['embedding_size']
  
  def __init__(self, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
    self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
    self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)

  @jit.script_method
  def forward(self, observation):
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    hidden = hidden.view(-1, 1024)
    hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
    return hidden


def Encoder(symbolic, observation_size, embedding_size, activation_function='relu'):
  if symbolic:
    return SymbolicEncoder(observation_size, embedding_size, activation_function)
  else:
    return VisualEncoder(embedding_size, activation_function)


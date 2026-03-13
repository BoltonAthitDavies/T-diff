# Diffusion Transformer Architecture

This document explains the diffusion-based trajectory model used in this repository for ObjectGoal Navigation. It is written to help you connect the paper idea to the implementation.

## 1. What the model predicts

The model predicts a short future trajectory segment instead of a single next action.

- Each predicted step is a 2D point in map coordinates.
- The full output is a sequence with shape `[B, horizon, 2]`.
- In the Gibson training config, `horizon=28` and the control loop consumes `n_action_steps=8` future points.

The main idea is that planning a sequence gives the agent a more stable and less myopic navigation signal than predicting only the next move.

## 2. High-level pipeline

At inference time the architecture works like this:

1. Build or update a semantic map from the current episode.
2. Encode the semantic map with a ResNet-18 backbone.
3. Concatenate map features with the current agent location and target-object one-hot vector.
4. Use those features as conditioning for a diffusion Transformer.
5. Start from Gaussian noise and iteratively denoise a trajectory sequence.
6. Take the predicted future points and pass the near-term segment to the planner.

In this repository, the runtime integration happens in `semexp/eval_tdiff.py` and the core policy implementation exists in two parallel codepaths:

- Training path: `train_traj/trajectory_diffusion/policy/trajectory_diffusion_transformer_gibson_policy.py`
- Evaluation path: `diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py`

The two implementations are nearly the same in structure.

## 3. Inputs to the model

The diffusion model is conditioned on three observation components:

### Semantic map

- Shape in config: `[19, 224, 224]`
- Type: image-like tensor
- Meaning: obstacle, explored area, agent traces, and semantic categories projected into a global or local map representation

### Goal category

- Shape in config: `[19]`
- Type: low-dimensional vector
- Meaning: one-hot representation of the target object category

### Current location

- Shape in config: `[2]`
- Type: low-dimensional vector
- Meaning: current 2D agent position in the map frame

These observation definitions are declared in `train_traj/train_diffusion_traj_gibson.yaml`.

## 4. Observation encoder

The image encoder is a modified ResNet-18.

- Base model: ResNet-18 pretrained on ImageNet
- First convolution changed from 3-channel input to 19-channel input
- Final fully connected layer removed so the network outputs a 512-dimensional feature vector

Implementation:

```python
self.obs_encoder = resnet_obs_encoder.resnet18(
    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
)
self.obs_encoder.conv1 = nn.Conv2d(19, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
self.obs_encoder.fc = nn.Sequential()
```

After encoding the map, the model builds a single conditioning token by concatenating:

- `loc`: 2 dimensions
- `target`: 19 dimensions
- ResNet feature: 512 dimensions

Total conditioning dimension:

$$2 + 19 + 512 = 533$$

That is why the policy sets `obs_feature_dim = 533` in the Gibson version.

## 5. Diffusion Transformer core

The denoising network is `TransformerForDiffusion`.

Its job is to predict either the added noise or the clean sample target, depending on the scheduler configuration. In this repo the config uses:

- DDPM scheduler
- `prediction_type: epsilon`

That means the model is trained to predict the noise that was added to the clean trajectory.

### Token structure

There are two streams in the Transformer:

1. Main sequence tokens
   - One token per trajectory timestep
   - Each token starts from a 2D noisy waypoint and is projected to the Transformer embedding dimension

2. Conditioning tokens
   - Diffusion timestep embedding
   - Observation conditioning token(s)

### Encoder-decoder layout

When observation conditioning is enabled, the Transformer uses:

- A condition encoder for the timestep and observation tokens
- A decoder over the noisy trajectory tokens
- Cross-attention from trajectory tokens to conditioning memory

This is implemented in `diffusion_policy/model/diffusion/transformer_for_diffusion.py` and mirrored under `train_traj/trajectory_diffusion/model/diffusion/`.

Conceptually:

```text
condition tokens = [time embedding, obs conditioning]
trajectory tokens = noisy future waypoints

encoded_condition = condition_encoder(condition tokens)
denoised_features = transformer_decoder(trajectory tokens, memory=encoded_condition)
predicted_noise = linear_head(denoised_features)
```

## 6. Training objective

Training follows the standard diffusion pattern:

1. Start with clean expert trajectory sequence.
2. Sample random Gaussian noise.
3. Sample a random diffusion timestep.
4. Add noise to the clean trajectory using the scheduler.
5. Ask the Transformer to predict the noise.
6. Optimize mean squared error between predicted and true noise.

The loss path in the policy is:

```python
noise = torch.randn(trajectory.shape, device=trajectory.device)
timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device).long()
noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
pred = self.model(noisy_trajectory, timesteps, cond)
loss = F.mse_loss(pred, noise, reduction='none')
```

This happens in `compute_loss()` inside `train_traj/trajectory_diffusion/policy/trajectory_diffusion_transformer_gibson_policy.py`.

## 7. Inference procedure

Inference is iterative denoising.

1. Create a random trajectory tensor sampled from Gaussian noise.
2. For each scheduler timestep:
   - apply any hard conditioning mask
   - run the Transformer to predict the noise residual
   - ask the scheduler to step from $x_t$ to $x_{t-1}$
3. Return the final denoised trajectory

In code this happens in `conditional_sample()`.

Conceptually:

```text
x_T ~ N(0, I)
for t = T ... 1:
    eps_hat = model(x_t, t, cond)
    x_{t-1} = scheduler.step(eps_hat, t, x_t)
return x_0
```

The policy then slices out the actionable near-term portion:

- full prediction: `action_pred`
- executed subset: `action = action_pred[:, start:end]`

## 8. Important config values

The main Gibson config is `train_traj/train_diffusion_traj_gibson.yaml`.

Key values:

- `horizon: 28`
- `n_action_steps: 8`
- `n_obs_steps: 1`
- `num_train_timesteps: 100`
- `num_inference_steps: 100`
- `n_layer: 8`
- `n_head: 4`
- `n_emb: 256`
- `causal_attn: True`
- `obs_as_cond: True`

These settings mean the model predicts a 28-step future 2D path while conditioning on one observation step.

## 9. Data flow through the repo

If you want to follow the architecture through the codebase, this is the shortest path:

1. `train_traj/train.py`
   - Hydra entrypoint for training
2. `train_traj/train_diffusion_traj_gibson.yaml`
   - model and training configuration
3. `train_traj/trajectory_diffusion/workspace/train_diffusion_transformer_gibson_workspace.py`
   - training loop, dataloaders, optimizer, EMA, checkpointing
4. `train_traj/trajectory_diffusion/policy/trajectory_diffusion_transformer_gibson_policy.py`
   - observation encoder, diffusion sampling, training loss
5. `train_traj/trajectory_diffusion/model/diffusion/transformer_for_diffusion.py`
   - Transformer encoder-decoder denoiser
6. `semexp/eval_tdiff.py`
   - evaluation-time integration into navigation

## 10. Why this architecture makes sense for ObjectNav

This model is a good fit for ObjectNav for three reasons:

1. It plans a sequence, not only the next move.
2. It conditions on a semantic map, which captures spatial structure better than a single RGB frame.
3. Diffusion handles multimodal futures better than direct one-shot regression.

In practice, this gives the planner a smoother and more globally coherent guide trajectory.

## 11. One implementation detail to remember

There are two similar policy stacks in this repository:

- `train_traj/trajectory_diffusion/...`
- `diffusion_policy/...`

For understanding the training setup described by the Gibson config, the `train_traj/trajectory_diffusion/...` path is the primary one. The `diffusion_policy/...` path contains closely related policy code used by the evaluation stack.

## 12. Short mental model

You can think about the model as:

- ResNet map encoder: compresses the semantic map
- Goal and location fusion: tells the model where it is and what to search for
- Diffusion Transformer: turns noisy waypoint sequences into plausible future paths
- Planner interface: uses the first few predicted points and replans repeatedly

That combination is the core of T-Diff in this repository.
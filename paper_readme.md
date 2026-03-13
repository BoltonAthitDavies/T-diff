# Trajectory Diffusion for ObjectGoal Navigation

## Overview
This README summarizes the NeurIPS 2024 paper **"Trajectory Diffusion for ObjectGoal Navigation"**. The paper introduces **T-Diff**, a diffusion-based sequence planner for the ObjectNav task. Instead of predicting only the next action or waypoint, T-Diff generates a **future trajectory segment** conditioned on the agent’s current semantic map and target object, aiming to improve temporal consistency and reduce myopic planning.

## Problem Setting
In **ObjectGoal Navigation (ObjectNav)**, an embodied agent must navigate to a target object category, such as a chair or toilet, in an unseen indoor environment using visual observations. Existing methods often rely on **single-step planning**, which can overlook long-term structure and produce short-sighted decisions.

The paper argues that navigation is naturally a **sequential decision-making problem**, so planning an entire future path should be more effective than deciding only the immediate next step.

## Main Contribution
The main contribution of the paper is **Trajectory Diffusion (T-Diff)**, a method that learns to generate a **sequence of future trajectory points** rather than a single-step subgoal. The generated trajectory is conditioned on:

- the agent’s current **semantic map**
- the **target object category**

This gives the agent a more temporally coherent and globally meaningful navigation plan.

## Method Summary
The overall pipeline is:

1. The agent receives **RGB-D observations** from the environment.
2. These observations are used to build and update a **semantic map**.
3. A diffusion model generates a **future trajectory segment** from the current map and target object.
4. A waypoint is selected from the generated trajectory.
5. A local planner moves the agent toward that waypoint.

### Model Design
- The semantic map is encoded using **ResNet-18**.
- The goal object is represented through an embedding.
- A **Transformer-based diffusion model** learns to denoise noisy trajectory samples into valid future trajectories.
- Training uses automatically collected **optimal trajectory segments** obtained from precise training maps.

## Why Diffusion?
The paper explains that directly learning the mapping from semantic map and goal to future trajectory is difficult because the conditional distribution is high-dimensional and sparse. A diffusion model helps by breaking this difficult generation task into a sequence of denoising steps, making learning more stable and expressive.

The authors also compare T-Diff with a direct decoder and show that the diffusion-based approach performs significantly better.

## Key Results
The method is evaluated on **Gibson** and **Matterport3D (MP3D)** using standard ObjectNav metrics:

- **SR**: Success Rate
- **SPL**: Success weighted by Path Length
- **DTS**: Distance To Success

### Reported Performance
| Dataset | SR (%) | SPL (%) | DTS (m) |
|--------|-------:|--------:|--------:|
| Gibson | 79.6 | 44.9 | 1.00 |
| MP3D | 39.6 | 15.2 | 5.16 |

The paper reports that T-Diff outperforms several prior end-to-end and modular baselines on both datasets.

## Important Findings
The experiments suggest that:

- **sequence planning** is more effective than single-step planning for ObjectNav
- conditioning on a **semantic map** is better than using only the current RGB observation
- including the **goal object** improves trajectory prediction
- predicting a **full future segment** leads to better navigation behavior
- T-Diff shows better **cross-simulator generalization** than some modular baselines such as PONI

## Strengths
- Clear motivation: navigation should benefit from temporally consistent sequence planning
- Combines **semantic spatial memory** with **diffusion-based generation**
- Strong benchmark performance on Gibson and MP3D
- Better scalability across different simulators

## Limitations
- Training depends on **optimal trajectories** collected from precise maps
- The method still uses a **local planner** for action execution
- Evaluation is conducted in simulation, so real-world deployment is not directly demonstrated
- Performance on MP3D, while improved, still shows that ObjectNav remains a challenging task

## Conclusion
This paper proposes a strong alternative to traditional single-step ObjectNav planning. By generating a future trajectory sequence with a diffusion model, T-Diff improves temporal consistency, guidance quality, and overall navigation performance. The work shows that **trajectory-level sequence planning** is a promising direction for embodied navigation.

## Citation
Yu, X., Zhang, S., Song, X., Qin, X., and Jiang, S. **Trajectory Diffusion for ObjectGoal Navigation.** NeurIPS 2024.

## Source
Based on the uploaded paper: *NeurIPS-2024-trajectory-diffusion-for-objectgoal-navigation-Paper-Conference.pdf*.


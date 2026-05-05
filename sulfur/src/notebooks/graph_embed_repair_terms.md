# Embedding-Based Node Repair: Term Glossary

## Core Objective Terms

- Node repair: Updating node input features so a frozen classifier changes wrong predictions into correct predictions.
- Frozen model: A model whose weights are not updated during optimization. Gradients can still flow through it to optimize inputs.
- Input-space repair: Repair done by changing node feature vectors directly, not by retraining model weights.
- Overfit mode: Intentionally optimizing strongly for the current graph labels, with weak concern for generalization.
- Full-label supervision: Using labels for all relevant nodes in the optimization objective.
- Train-only supervision: Using labels only for train-mask nodes in the optimization objective.

## Graph/Data Terms

- Node: A vertex in the graph, indexed by integer ID.
- Edge index: A tensor with shape [2, E] representing E directed edges.
- Feature matrix X: Tensor with shape [N, F], where N is number of nodes and F is feature dimension.
- Label vector y: Tensor with shape [N], where each entry is the class index for one node.
- Train/val/test masks: Boolean vectors of shape [N] defining split membership.
- Target nodes: Nodes selected for repair optimization.
- Non-target nodes: Nodes not selected for direct repair updates.

## Embedding/Optimization Terms

- Embedding table: A trainable parameter matrix indexed by node ID.
- Node embedding row: One row in the embedding table corresponding to one node.
- Delta parameter: Trainable per-node additive offset used to modify original features.
- Masked update: Applying updates only on selected node rows using a boolean mask.
- Gradient flow: Backpropagated derivatives from loss to trainable parameters.
- Gradient norm: Magnitude of gradient, typically used to verify optimization signal.
- Zero-gradient row: A row receiving effectively no optimization signal for a step.
- Learning rate (LR): Step size used by optimizer to update parameters.
- Scheduler: Rule for adapting LR during optimization.
- Regularization: Penalty term to constrain updates (for example, L2 distance to original features).

## Prediction/Uncertainty Terms

- Logits: Raw model outputs before softmax, shape [N, C].
- Softmax probabilities: Class probabilities derived from logits.
- Predicted class: Argmax of probability/logit per node.
- Correct prediction mask: Boolean vector indicating whether prediction equals ground truth.
- Incorrect nodes: Nodes with wrong baseline predictions.
- Confidence: Max softmax probability for a node.
- Entropy: Uncertainty measure from class probabilities, higher means less certain.
- High-confidence incorrect node: Wrong node with high max probability.

## Repair Outcome Terms

- Fixed node: Node that changed from wrong to correct after repair.
- Regressed node: Node that changed from correct to wrong after repair.
- Net gain: Fixed count minus regressed count.
- Still-wrong node: Node that remains incorrect after repair.
- Always-correct node: Node correct before and after repair.
- Coverage: Fraction of nodes affected by a selection or trust rule.

## Drift/Distance Terms

- Feature drift: Difference between repaired and original feature vectors.
- Per-node L2 drift: Euclidean norm of the per-node feature difference.
- Per-node cosine distance: 1 minus cosine similarity between repaired and original feature vectors.
- Drift distribution: Histogram/CDF/violin summary of per-node drift magnitudes.
- Drift stratification: Grouped drift analysis by outcomes (fixed, regressed, etc.).

## Selection and Neighborhood Terms

- Entropy-ranked selection: Choosing nodes by sorted entropy values.
- Budget: Number of nodes selected for repair in one run.
- k-hop neighborhood: Nodes within graph distance <= k from a seed set.
- Affected set: Union of selected nodes and optional neighborhood expansion.
- Target-only scope: Repair updates restricted to selected nodes only.
- k-hop scope: Repair updates applied to selected nodes and their neighborhood.

## Trust/Counterfactual Terms

- Reconstruction error: MSE between reconstructed and original features.
- OOD flag: Heuristic indicator that a node looks out-of-distribution.
- Trust score: Scalar confidence proxy, often inverse of reconstruction error.
- Abstention: Marking a prediction as not trusted instead of returning a class.
- Counterfactual witness: A graph edit (for example edge removal) that flips a wrong prediction to correct.
- Witness gain: Increase in true-class probability under a counterfactual edit.

## Evaluation/Visualization Terms

- Baseline: Metrics computed before any repair optimization.
- Post-repair: Metrics computed after optimization.
- Split accuracy: Accuracy computed on train/val/test masks separately.
- Repair trajectory: Metrics tracked over epochs.
- Fixed-regressed Pareto view: Plot comparing gains vs collateral damage.
- Entropy-drift scatter: Plot relating uncertainty and feature movement.
- Outcome-colored graph view: Subgraph plot with colors by fixed/regressed/etc.

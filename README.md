# Distance-3 Surface Code GNN Decoder — Training Report

- Dataset: Rotated planar code, distance 3 (9 data qubits, 8 syndromes).
- Error model: i.i.d. bit flips with probability $p = 0.05$.
- Samples: 10,000 train / 2,000 test.
- Training: 20 epochs, batch 64, AdamW (lr $10^{-4}$, cosine anneal), weight decay $10^{-4}$.
- Results:
  - Correction rate (test): **95.8%**
  - Valid input rate (already zero syndrome): **62.5%**
  - Effective correction on nontrivial cases: **88.8%**
  - Loss at end: **~0.006**
  - Plot: `training_curves.png` saved by the script.

## Code Architecture (surface_code_decoder_d3.py)
- Data generation (panqec RotatedPlanar2DCode): builds $H_X \in \{0,1\}^{4\times 9}$ and $H_Z \in \{0,1\}^{4\times 9}$ parity-check matrices; syndromes are $s_X = H_X e \bmod 2$, $s_Z = H_Z e \bmod 2$.
- Graph: 8 syndrome nodes connected as a 1D chain with self-loops (easy message passing baseline).
- Model (`SurfaceCodeGNN`):
  - Embed syndrome bits (scalar $\to$ $d$) per node.
  - Three GCNConv layers with residual MLP refinement.
  - Per-iteration MLP head outputs qubit error probabilities (length 9); sigmoid activations.
- Training objective (sound inverse problem):
  - Soft predictions $\hat e \in [0,1]^9$ are pushed so their real-valued syndromes match inputs:
    $$L = \frac{1}{T} \sum_{t=1}^T w_t\big(\lVert H_X \hat e_t - s_X \rVert_2^2 + \lVert H_Z \hat e_t - s_Z \rVert_2^2\big)$$
    with later iterations weighted more ($w_t = t/T$) and $T=3$.
  - Rationale: if $H\hat e = s$, then $(e + \hat e)H^\top = s + s = 0$ (mod 2), i.e., the model learns an inverse map from syndromes to a canceling correction without ever seeing true errors.
- Evaluation: take final head, round $\hat e$ to $\{0,1\}$, apply $e' = (e + \hat e) \bmod 2$, recompute syndromes via $H_X, H_Z$; success if both vanish.
- Metrics tracked per epoch: training loss, correction rate; plot saved at end.

## Mathematical Notes
- Parity-check structure: for distance-3 rotated planar, $H_X$ and $H_Z$ each have 4 weight-4 checks; total 8 syndrome bits. Data qubits = 9, so the nullspace has dimension 5 and many low-weight errors are undetectable, explaining the 62.5% trivial-syndrome rate at $p=0.05$.
- Continuous relaxation: training uses real arithmetic (no $\bmod 2$) so gradients flow; binarization only at inference. This approximates solving the linear inverse $H \hat e \approx s$ under $[0,1]$ constraints, analogous to least-squares decoding.
- Effective correction: with 62.5% of cases already clean, the 95.8% overall rate implies $88.8\%$ success on the 37.5% nontrivial cases: $$\text{eff} = \frac{0.958 - 0.625}{1 - 0.625} \approx 0.888.$$

## Files of Interest
- Training/eval + plotting: [surface_code_decoder_d3.py](surface_code_decoder_d3.py)
- Generated plot: training_curves.png (created after running the script)

## Next Steps
- Try richer graph topology (2D layout of stabilizers) to exploit geometry.
- Sweep $p$ values and distance for robustness curves.
- Add calibration: temperature scaling of logits before rounding, or use a small search (e.g., local flips) guided by $H \hat e$ residuals.

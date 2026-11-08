# The additive ambiguity of gravitational entropy fixed by the topology of the horizon

Verification code for the paper

> S. A. Tsilioukas, *The additive ambiguity of gravitational entropy fixed by the topology of the horizon*, arXiv:XXXX.XXXXX [gr-qc] (2026).

The paper argues that in four-dimensional Einstein–Gauss–Bonnet gravity the additive constant of algebraic (type II) generalized entropy is the Gauss–Bonnet contribution to the Wald entropy, $2\pi\alpha\chi(\Sigma)/G$, and that it sits in the normalization of the trace and not in the state. The scripts in this repository reproduce the symbolic and numerical checks quoted in the text.

## Requirements

Python 3.10 or later with `sympy`, `numpy`, `scipy` and `mpmath`:

```bash
python -m pip install -r requirements.txt
```

The stored outputs were produced with Python 3.13, SymPy 1.14.0, NumPy 2.5, SciPy 1.18 and mpmath 1.3.

## Usage

Each script runs on its own, for example

```bash
cd scripts
python 03_test1_conical_expansion.py
```

To run everything (about 90 seconds on a laptop) and compare with the stored outputs in `expected_output/`:

```bash
python run_all.py --check
```

Running `python run_all.py` without `--check` regenerates `expected_output/`.

## Contents

| Script | Paper | What it checks |
|---|---|---|
| `scripts/01_table1_compact_manifolds.py` | Table I, App. A (i)–(ii) | Curvature invariants and $\int\mathcal{G}\sqrt{g}=32\pi^2\chi$ for $S^4$ and $S^2\times S^2$ |
| `scripts/02_table1_euclidean_schwarzschild.py` | Table I, Sec. III, App. A (iii) | $\int\mathcal{G}\sqrt{g}=64\pi^2$, $\oint{}^{(2)}\!R\,dA=8\pi$, Gauss–Bonnet Wald entropy $4\pi\alpha/G$ |
| `scripts/03_test1_conical_expansion.py` | Sec. VI (Test 1) | Exact linearity of the conical term in $1-n$; two tip regularizations, $a=0.1,1,5$ |
| `scripts/04_radial_ansatz_artifact.py` | Sec. VI.C | Total-derivative form of Eq. (conicalresult) for arbitrary $h,b$; the factor $1-b'(0)^2$ comes from a curvature singularity of the radial ansatz |
| `scripts/05_squashed_cone_fps.py` | Sec. III.D, VI.C | The Fursaev–Patrushev–Solodukhin defects combine into the intrinsic curvature of the cut (Gauss–Codazzi, generic metric) |
| `scripts/06_test2_noether_binormal.py` | Sec. VII.A, VII.D | Binormal contraction of $\partial\mathcal{L}/\partial R_{abcd}$ on Schwarzschild; the Gauss–Bonnet charge drops out of the first law |
| `scripts/07_test2_rigidity_kerr.py` | Sec. VII.B | $\oint K\,dA=4\pi$ on strongly deformed cuts and on the Kerr horizon ($a/M\le0.99$); torus control |
| `scripts/08_test2_trace_rescaling.py` | Sec. VII.C | Trace rescaling in a discretized crossed product: $\Delta S=\log\lambda$, state independent; expectation values and relative entropies unchanged |
| `scripts/09_lanczos_random_curvature.py` | Sec. VIII.B, VIII.G | The Lanczos tensor vanishes identically in $D=4$ and not in $D=5,6$; $\mathcal{G}=0$ on $S^1\times S^3$ |
| `scripts/10_jt_disk.py` | Sec. XI.B, App. A (0) | JT disk entropy, $\partial S/\partial S_0=1=\chi_{\rm disk}$, independent of $\beta$ |
| `scripts/11_jt_matter.py` | Sec. XI.E–F (Test 3) | Two-dimensional Gauss–Bonnet, matter renormalization of $S_0$, modular-flow test |
| `scripts/12_ads_topological_black_holes.py` | Sec. XII | Hyperbolic entropy $\pi(g-1)(r_h^2-4\alpha)/G$, first law within a sector, Hawking–Page shift |
| `scripts/corner/r1_corner_checks.py` | Sec. VIII.C–D | Euler corner form, Gauss equation, Wald/Jacobson–Myers normalization, transverse-traceless counterexample $4c\,\mathrm{Area}$, corner cancellation* |
| `scripts/corner/r1_transgression_checks.py` | Sec. VIII.D | Chern–Weil transgression identities, completed form equal to zero, Myers normalization |
| `scripts/corner/kerr_dilatation_check.py` | Sec. VIII.C (i) | The bare Euler corner form vanishes on the Kerr family |
| `scripts/corner/kerr_hajicek.py` | Sec. VIII.C | Hájíček one-form on Kerr horizon cuts (supporting check) |
| `scripts/corner/s1_stretched_horizon.py` | Sec. VIII.E | Stretched-horizon identities for any $f$ and $k$; completed exterior Euler charge vanishes; Euler number of the excised tip |
| `scripts/corner/s2_lens_joint.py` | Sec. VIII.E, Eq. (joint) | Joint term as the rounded-edge limit of the Myers term; lens test $32\pi^2$ for every opening angle |
| `scripts/corner/s3_gauge.py` | Sec. VIII.E | Lorentz edge charges of the bare form; the pairing vanishes on the stabilizers of $u$ and of a null vector, not on $\mathrm{sim}(2)$ |
| `scripts/corner/s4_triangle.py` | Sec. VIII.E | Triangle identity for transgressions with three references |

Helper modules: `scripts/curvature.py` (curvature invariants from a metric) and `scripts/corner/forms.py` (Lorentz-algebra-valued differential forms).

\* The last three entries printed by `r1_corner_checks.py` (compact BF level, growth mismatch, normalization arithmetic) support the companion paper on the index conjecture.

## Citation

If you use this code, please cite the paper above and the archived software release (see `CITATION.cff`; the Zenodo DOI is added at release).

## License

MIT, see `LICENSE`.

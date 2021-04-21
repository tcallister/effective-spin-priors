# effective-spin-priors

This repository is a companion to the technical note [A Thesaurus for Common Priors in Gravitational-Wave Astronomy](https://arxiv.org/abs/2104.09508), containining some helper functions meant to aid in calculating the priors `p(chi_effective|q)` and `p(chi_p|q)` on the effective inspiral spin `chi_effective` and effective precessing spin `chi_p` under a few common choices of component spin priors.

The file `priors.py` defines the following following functions:
* `chi_effective_prior_from_aligned_spins(q,a_max,chi_effectives)`: Returns the condition priors `p(chi_effectives|q,a_max)` implied by uniform and aligned component spin priors
* `chi_effective_prior_from_isotropic_spins(q,a_max,chi_effectives)`: Returns the condition priors `p(chi_effectives|q,a_max)` implied by uniform and isotropic component spin priors
* `chi_p_prior_from_isotropic_spins(q,a_max,chi_ps)`: Returns the condition priors `p(chi_ps|q,a_max)` implied by uniform and isotropic component spin priors

Usage of these functions is demonstrated in `Demo.ipynb`.


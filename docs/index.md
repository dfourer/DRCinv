# DRCinv Documentation

DRCinv is a Python library for feedforward dynamic range compression, expansion, and model-based inversion of audio signals.

The library provides two main functions:

- `drc_forward`: apply downward compression or downward expansion.
- `drc_inverse`: reconstruct the original signal from processed audio using model-based inversion.

## Installation

Install DRCinv from PyPI:

```bash
pip install drcinv
```

## Documentation

- [DRC Model](model.md)
- [Model-Based Inversion](inversion.md)
- [API Reference](api.md)
- [Examples](examples.md)

## Citation

If you use DRCinv in your research, please cite the following preprint:

H. Sun, D. Fourer, and H. Maaref, **"Neural-Enhanced Dynamic Range Compression Inversion: A Hybrid Approach for Restoring Audio Dynamics,"** *arXiv:2411.04337*, 2024.

**Preprint:** [arXiv:2411.04337](https://arxiv.org/abs/2411.04337)

**Publication status:** A related manuscript is currently under review at *IEEE/ACM Transactions on Audio, Speech, and Language Processing* (minor revision).

### BibTeX

```bibtex
@misc{sun2024neural,
  title={Neural-Enhanced Dynamic Range Compression Inversion: A Hybrid Approach for Restoring Audio Dynamics},
  author={Sun, Haoran and Fourer, Dominique and Maaref, Hichem},
  year={2024},
  eprint={2411.04337},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  doi={10.48550/arXiv.2411.04337}
}
```

*Note: This reference is provisional and will be updated upon publication.*

## Project links

- [GitHub repository](https://github.com/dfourer/DRCinv)
- [PyPI](https://pypi.org/project/drcinv/)
- [Research project](https://hdrc-inv.github.io/)

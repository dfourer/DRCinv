# API Reference

This page provides the API documentation for the two main functions
of DRCinv.

## Forward Dynamic Range Control

Apply feedforward downward compression or downward expansion
to an input audio signal.

::: drcinv.drc_forward
    options:
      show_root_heading: true
      show_source: false

## Model-Based Inversion

Reconstruct the original audio signal from the output of a
downward compressor or downward expander.

::: drcinv.drc_inverse
    options:
      show_root_heading: true
      show_source: false

## Parameter Estimation

The `drcinv.drcparam` namespace provides neural and black-box estimators.

::: drcinv.drcparam.MEEParameterEstimator
    options:
      show_root_heading: true
      show_source: false

::: drcinv.drcparam.BlackBoxParameterEstimator
    options:
      show_root_heading: true
      show_source: false

::: drcinv.drcparam.estimate_drc_parameters
    options:
      show_root_heading: true
      show_source: false

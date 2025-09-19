"""
AI Steerability 360 toolkit.

The AI Steerability 360 toolkit (AISteer360) enables systematic control over language model behavior through four model
control surfaces: input, structural, state, and output. Methods can be composed into composite model operations (via
steering pipelines). Benchmarks enable comparison of steering pipelines on common use cases.
"""

try:
    from .version import version as __version__
except ImportError:
    pass

"""G1-ecosystem motion library -> Dropbear motion CSV pipeline (CPU only, numpy).

Modules
-------
* :mod:`.g1_sources`       loaders for all on-disk G1 clip formats -> :class:`~.g1_sources.G1Motion`
* :mod:`.g1_model`         G1 joint orders + pure-numpy MJCF forward kinematics
* :mod:`.g1_to_dropbear`   G1 -> Dropbear semantic trajectory -> motors + root (``world`` body) pose
* :mod:`.calibration_view` access to ``dropbear-semantic-calibration-v1`` + the contract SemanticMap
* :mod:`.motion_csv`       ``dropbear-motion-csv-v1`` writer / reader / validator
* :mod:`.synthetic`        Dropbear-native synthetic clips (stand, wave, arm swing, weight shift, squat)
* :mod:`.pipeline`         file-level retarget used by the CLI tools
"""

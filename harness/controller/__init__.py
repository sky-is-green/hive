"""The Hive settings controller: hardware profile, policy, and encoder seam.

The controller is deliberately split into layers:

* :mod:`harness.controller.profile` turns hardware probes into a saved snapshot;
* :mod:`harness.controller.recommend` is the deterministic E0 policy;
* :mod:`harness.controller.encoder` is the lazy text-encoder boundary;
* :mod:`harness.controller.service` persists proposals and applies them.

No layer imports a model at import time, and no layer writes launch flags that
are not in the controller allowlist.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"

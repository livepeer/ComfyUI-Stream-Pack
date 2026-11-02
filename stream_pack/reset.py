"""Clear per-customer state inside the owning workflow process."""

import sys


def scrub_worker():
    module = sys.modules.get("stream_pack.legacy.flux_klein_stream")
    if module is not None:
        module.reset_session()

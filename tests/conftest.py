"""
conftest.py for brainglobe-heatmap test suite.

Sets the matplotlib backend to a non-interactive one (Agg) before any test
module is collected or imported. This must happen here — at the earliest
possible point in pytest's startup — rather than inside individual test files,
because ``matplotlib.use()`` has no effect once ``matplotlib.pyplot`` has
already been imported.

Background
----------
On Windows Server 2025 GitHub Actions runners, the Python installation
sometimes ships with a broken or missing ``tcl/init.tcl``, causing the
default TkAgg backend to raise::

    _tkinter.TclError: Can't find a usable init.tcl in the following
    directories: {C:\\...\\tcl\\tcl8.6}

The failure is sporadic because it depends on runner provisioning state.
Forcing Agg -- a non-interactive, file-rendering-only backend -- removes
the Tk/Tcl dependency entirely and makes the test suite deterministic
across all platforms and CI environments.

See: https://github.com/brainglobe/brainglobe-heatmap/issues/98
"""

import matplotlib

matplotlib.use("Agg")

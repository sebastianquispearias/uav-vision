"""Lets `pytest` run this repository's gates without rewriting a single one of them.

The gates are deliberately plain scripts and not pytest functions: each one prints its numbers
and ends in TODO OK, so it can be read as a measurement and run on a Raspberry with nothing
installed. That is worth keeping, and it is also the reason a bare `pytest` finds nothing here:
there are no test functions to collect.

So each gate is collected as ONE pytest item and run as a subprocess. It passes on exit code 0,
and it is SKIPPED, not failed, when what is missing is an optional dependency or the companion
flight archive: a laptop without boxmot has not broken anything, and a red test that only means
"not installed here" is a red test people learn to ignore.

The .js gates of the ground station are collected the same way when node is on the PATH.
"""
import os
import shutil
import subprocess
import sys

import pytest

RAIZ = os.path.dirname(os.path.abspath(__file__))

FALTA = (
    "No module named 'boxmot'",
    "No module named 'open_clip'",
    "No module named 'torch'",
    "No module named 'selenium'",
    "No module named 'cv2'",
    "No module named 'gradys_embedded'",
    "No module named 'sklearn'",
    "No module named 'scipy'",
    "sin repo hermano",
)


def pytest_pycollect_makemodule(module_path, parent):
    """Replaces pytest's own Module collector for a gate, so the gate is never IMPORTED.

    This hook and not pytest_collect_file, and the difference matters: collect_file is not a
    firstresult hook, so returning a node there ADDS one and pytest still builds its Module
    beside it. It then imports the gate into its own process, where a gate that skips itself
    with sys.exit(0) surfaces as a collection error.
    """
    if module_path.parent.name == "tests" and module_path.name.startswith("test_"):
        return ArchivoGate.from_parent(parent, path=module_path)
    return None


def pytest_collect_file(parent, file_path):
    """Collects the ground station's .js gates, which pytest knows nothing about."""
    if (file_path.parent.name == "tests" and file_path.name.startswith("test_")
            and file_path.suffix == ".js" and shutil.which("node")):
        return ArchivoGate.from_parent(parent, path=file_path)
    return None


class ArchivoGate(pytest.File):
    def collect(self):
        yield Gate.from_parent(self, name=self.path.name)


class Gate(pytest.Item):
    def runtest(self):
        orden = ([sys.executable] if self.path.suffix == ".py" else ["node"]) + [str(self.path)]
        r = subprocess.run(orden, cwd=RAIZ, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        salida = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            for falta in FALTA:
                if falta in salida:
                    pytest.skip("no esta disponible aqui: %s" % falta)
            if "FileNotFoundError" in salida and "drone-geolocation" in salida:
                pytest.skip("necesita el archivo de vuelos, que no viaja en este repo")
            raise GateFallo(salida)

    def repr_failure(self, excinfo, style=None):
        if isinstance(excinfo.value, GateFallo):
            return excinfo.value.args[0]
        return super().repr_failure(excinfo, style=style)

    def reportinfo(self):
        return self.path, 0, self.path.name


class GateFallo(Exception):
    """A gate that ran and did not pass. Its own output is the whole report."""

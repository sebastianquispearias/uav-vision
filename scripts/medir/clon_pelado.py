"""Runs anything as if this repository had just been cloned on its own, with nothing beside it.

A gate that passes on the author's laptop says nothing about what somebody gets when they clone:
the laptop has the companion repositories next door, the flight archive on disk and every model
downloaded. This answers the question the laptop cannot.

    python scripts/medir/clon_pelado.py python demo/demo.py --sin-mapa
    python scripts/medir/clon_pelado.py python -m pytest -q

It works by putting a sitecustomize on PYTHONPATH that BLOCKS the import of the sibling runtime
and any read under the sibling repositories, so the child process sees what a clone sees.

BLOCKING THE IMPORT AND NOT MOVING THE DIRECTORIES IS THE POINT. The obvious way is to rename
the sibling repositories out of the way, and on 2026-10-04 that was tried and `mv` failed with
"Device or resource busy" -- while the loop around it happily reported 50 of 50 passing, because
the siblings were still there. The signal was not measuring what it looked like it measured.
Blocking the import does not depend on the filesystem cooperating.

What it found the first time it ran: the first command of the README did not work, and 14 of the
50 gates needed a repository the README mentioned only in passing, 170 lines below.
"""
import os
import subprocess
import sys
import tempfile

CORTAFUEGOS = '''
"""Makes a process see what a clone of uav_vision alone sees."""
import builtins
import sys

BLOQUEADOS = ("gradys_embedded", "gradysim", "gradys_core")
AJENOS = ("drone-geolocation", "gradys-embedded", "gradys-sim")


class _Cortafuegos:
    def find_module(self, nombre, ruta=None):
        return self.find_spec(nombre, ruta)

    def find_spec(self, nombre, ruta=None, destino=None):
        if nombre.split(".")[0] in BLOQUEADOS:
            raise ImportError("sin repo hermano: %s no esta en un clon pelado" % nombre)
        return None


sys.meta_path.insert(0, _Cortafuegos())
_abrir = builtins.open


def _open(archivo, *a, **k):
    texto = str(archivo).replace("\\\\", "/")
    if any(x in texto for x in AJENOS):
        raise FileNotFoundError("sin repo hermano: %s" % archivo)
    return _abrir(archivo, *a, **k)


builtins.open = _open
'''


def main(orden):
    if not orden:
        print(__doc__.strip())
        return 2
    carpeta = tempfile.mkdtemp(prefix="clon_pelado_")
    with open(os.path.join(carpeta, "sitecustomize.py"), "w", encoding="utf-8") as fh:
        fh.write(CORTAFUEGOS)
    entorno = dict(os.environ)
    anterior = entorno.get("PYTHONPATH")
    entorno["PYTHONPATH"] = carpeta + (os.pathsep + anterior if anterior else "")
    print("-- como un clon pelado: sin gradys_embedded y sin leer los repos de al lado",
          flush=True)
    return subprocess.run(orden, env=entorno).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

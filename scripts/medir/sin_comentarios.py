"""Removes loose comments from a Python file, by tokenizer and not by regex.

The tokenizer is the point: a '#' inside a string literal or a regex is not a COMMENT token, so
it can never be mistaken for one. A blanket regex eats the CSS colours of an embedded page and
leaves a file that still compiles, which is the worst kind of failure.

Two kinds of comment survive:

  - DIRECTIVES the tooling reads: noqa, type:, pragma, fmt:, pylint, mypy, ruff, nosec, the
    encoding declaration and the shebang. They are comments to Python but not to a human.
  - Whatever --conservar names, as a comma-separated list of substrings. This is the PEP 8
    escape hatch: an inline comment stays when the WHY of the line cannot be recovered from the
    code, such as an algebraic identity, a sign convention, a library quirk or a magic index
    into somebody else's format.

Usage:
    python sin_comentarios.py <file.py> [more.py ...]                 report only
    python sin_comentarios.py --aplicar <file.py> [...]               rewrite in place
    python sin_comentarios.py --aplicar --conservar='R.T inverts,COCO ids' <file.py>
"""
import re
import sys
import tokenize

DIRECTIVA = re.compile(r'^#\s*(noqa|type:|pragma|fmt:|pylint|mypy|ruff|nosec|-\*-|!)', re.I)


def comentarios(ruta):
    """Every COMMENT token in the file, as (line, column, text)."""
    fuera = []
    with open(ruta, 'rb') as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type == tokenize.COMMENT:
                fuera.append((tok.start[0], tok.start[1], tok.string))
    return fuera


def limpiar(ruta, aplicar=False, conservar=()):
    """Drops the comments that are neither a directive nor named in `conservar`."""
    def se_queda(texto):
        if DIRECTIVA.match(texto):
            return True
        return any(c and c in texto for c in conservar)

    com = comentarios(ruta)
    quitar = [c for c in com if not se_queda(c[2])]
    guardar = [c for c in com if se_queda(c[2])]
    if not aplicar:
        return len(quitar), len(guardar)

    lineas = open(ruta, encoding='utf-8').read().split('\n')
    por_linea = {n: col for n, col, _ in quitar}
    salida = []
    for i, l in enumerate(lineas, 1):
        if i in por_linea:
            recortada = l[:por_linea[i]].rstrip()
            if recortada == '':
                continue
            salida.append(recortada)
        else:
            salida.append(l)
    open(ruta, 'w', encoding='utf-8', newline=chr(10)).write('\n'.join(salida))
    return len(quitar), len(guardar)


if __name__ == '__main__':
    args = sys.argv[1:]
    aplicar = '--aplicar' in args
    conservar = ()
    for a in list(args):
        if a.startswith('--conservar='):
            conservar = tuple(x for x in a.split('=', 1)[1].split('||') if x)
            args.remove(a)
    rutas = [a for a in args if not a.startswith('--')]
    tq = tg = 0
    for r in rutas:
        q, g = limpiar(r, aplicar, conservar)
        tq += q
        tg += g
        if q or g:
            print("%-52s quita %3d   conserva %d" % (r, q, g))
    print("TOTAL: %d comentarios fuera, %d conservados" % (tq, tg))

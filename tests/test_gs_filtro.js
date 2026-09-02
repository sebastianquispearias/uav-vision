/*
 * Gate for the ground station page, without a browser.
 *
 * gs_mapa.py carries ~200 lines of JavaScript inside a Python string, and until this file
 * existed nothing checked any of it: a typo in the filter shipped silently and only showed up
 * as a page that had stopped drawing, in the field, with the drone in the air.
 *
 * This does not re-implement anything. It pulls the real <script> block out of gs_mapa.py,
 * stubs the few DOM calls the page makes, and runs the page's OWN pintar() / pintarFiltro() /
 * pintarLista() against a synthetic report. If the page changes, this runs the change.
 *
 * Run with: node tests/test_gs_filtro.js        (from the uav_vision root)
 */
'use strict';
const fs = require('fs');
const path = require('path');

// A path can be passed in to run this against a modified copy -- which is how the gate itself
// gets checked: point it at a deliberately broken page and it must fail.
const GS = process.argv[2] || path.join(__dirname, '..', 'scripts', 'banco_embedded', 'gs_mapa.py');
const py = fs.readFileSync(GS, 'utf8');
const html = /PAGINA = r"""([\s\S]*?)"""/.exec(py)[1];
let codigo = /<script>([\s\S]*?)<\/script>/.exec(html)[1];
// The last three lines boot the page against a live server and a live canvas. Everything above
// them is pure logic and is what we want to exercise.
codigo = codigo.replace(/redimensionar\(\);\s*refrescar\(\);\s*setInterval\(refrescar,\s*1000\);/, '');

// -- the smallest DOM the page will accept -----------------------------------
// Every canvas call is swallowed: this checks which POIs survive the filter, not pixels.
const nulo = new Proxy(function () {}, { get: () => nulo, apply: () => nulo });
const elems = {};
function elem(id) {
  if (!elems[id]) elems[id] = {
    id, innerHTML: '', textContent: '', className: '', style: {}, dataset: {},
    // the browser would parse the buttons the page just wrote; this reads them back out
    querySelectorAll() {
      return [...this.innerHTML.matchAll(/data-c="([^"]+)" class="([^"]*)"/g)]
        .map(m => ({ dataset: { c: m[1] }, clase: m[2], onclick: null }));
    },
    getBoundingClientRect: () => ({ width: 800, height: 600 }),
    getContext: () => nulo,
  };
  return elems[id];
}
global.document = { getElementById: elem };
global.window = { devicePixelRatio: 1, addEventListener: () => {} };

function ok(cond, msg) {
  if (!cond) { console.error('FALLO: ' + msg); process.exit(1); }
}

// The checks run inside the same eval as the page code: `estado`, `ocultas` and `visibles` are
// let-bound in that scope and are not reachable from out here.
const prueba = `
estado = { pois: [
  { x: 1, y: 2, cls: 'person', mature: true,  mobile: false, n_obs: 90, conf: .8, dron: 7 },
  { x: 5, y: 6, cls: 'car',    mature: true,  mobile: false, n_obs: 40, conf: .7, dron: 7 },
  { x: 9, y: 1, cls: 'car',    mature: false, mobile: false, n_obs: 12, conf: .5, dron: 7 },
  { x: 2, y: 8, cls: null,     mature: true,  mobile: true,  n_obs: 30, conf: .6, dron: 7 },
]};
pintar();
const botones = () => document.getElementById('filtro').querySelectorAll().map(b => b.dataset.c);
console.log('  sin filtro         :', document.getElementById('cuenta').textContent,
            '| botones:', botones().join(', '));
ok(botones().join(',') === 'car,person,sin clase', 'los botones no son las clases recibidas');
ok(document.getElementById('lista').innerHTML.indexOf('chip clase">car<') >= 0,
   'la tarjeta del coche no lleva chip de clase');

ocultas.add('car');
pintar();
console.log('  ocultando "car"    :', document.getElementById('cuenta').textContent,
            '| dibujados:', visibles.map(p => p.cls).join(', '));
ok(visibles.length === 2, 'ocultar car deberia dejar 2 POI de 4');
ok(document.getElementById('cuenta').textContent === '2 POI de 4',
   'la cabecera tiene que decir cuantos se estan escondiendo');
ok(document.getElementById('filtro').querySelectorAll()
     .find(b => b.dataset.c === 'car').clase === 'off', 'el boton oculto no queda tachado');

// A report with only cars in it. The person button must NOT disappear: it would take the
// operator's filter with it, silently, the moment a target left the frame.
estado = { pois: [{ x: 5, y: 6, cls: 'car', mature: true, mobile: false, n_obs: 40, conf: .7, dron: 7 }] };
pintar();
console.log('  solo llegan coches :', document.getElementById('cuenta').textContent,
            '| botones:', botones().join(', '));
ok(botones().indexOf('person') >= 0, 'el boton de una clase que dejo de llegar desaparecio');

ocultas.delete('car');
pintar();
console.log('  reactivando "car"  :', document.getElementById('cuenta').textContent);
ok(visibles.length === 1, 'reactivar la clase no devolvio el POI');
`;

console.log('======================================================================');
console.log('FILTRO DE CLASE DE LA GROUND STATION (codigo real de gs_mapa.py)');
console.log('======================================================================');
eval(codigo + prueba);
console.log();
console.log('TODO OK');

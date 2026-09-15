/*
 * Gate for how far the "limpiar" button reaches on the ground station page, without a browser.
 *
 * Clearing records where each non-persistent contact was, and a contact at such a place stays hidden
 * until it is seen again. Whether a POI sits at a cleared place used to be judged within the larger
 * of the cleared contact's radius and the POI's own 95 % radius -- the same mistake the operator's
 * verdicts had. A mobile POI carries a radius of hundreds of metres, so a contact that appeared after
 * the click, and was never cleared, was hidden anyway because its own circle covered the old place.
 *
 * The contrast: after clearing a lone doubtful point, a new mobile contact 20 m away must be drawn
 * while the cleared point stays hidden.
 *
 * Same harness as test_gs_veredicto_radio.js: the real <script> block of gs_mapa.py, a stub DOM, and
 * the page's own pintar() / limpiar() against synthetic reports.
 *
 * Run with: node tests/test_gs_limpiar_radio.js        (from the uav_vision root)
 */
'use strict';
const fs = require('fs');
const path = require('path');

const GS = process.argv[2] || path.join(__dirname, '..', 'scripts', 'banco_embedded', 'gs_mapa.py');
const py = fs.readFileSync(GS, 'utf8');
const html = /PAGINA = r"""([\s\S]*?)"""/.exec(py)[1];
let codigo = /<script>([\s\S]*?)<\/script>/.exec(html)[1];
codigo = codigo.replace(/redimensionar\(\);\s*refrescar\(\);\s*setInterval\(refrescar,\s*1000\);/, '');

const nulo = new Proxy(function () {}, { get: () => nulo, apply: () => nulo });
const elems = {};
function elem(id) {
  if (!elems[id]) elems[id] = {
    id, innerHTML: '', textContent: '', className: '', style: {}, dataset: {},
    querySelectorAll() {
      return [...this.innerHTML.matchAll(/data-c="([^"]+)" class="([^"]*)"/g)]
        .map(m => ({ dataset: { c: m[1] }, clase: m[2], onclick: null }));
    },
    getBoundingClientRect: () => ({ width: 800, height: 600 }),
    getContext: () => nulo,
    addEventListener() {},
  };
  return elems[id];
}
global.document = { getElementById: elem };
global.window = { devicePixelRatio: 1, addEventListener: () => {} };
global.setInterval = () => 0;
global.fetch = () => new Promise(() => {});

function ok(cond, msg) {
  if (!cond) { console.error('FALLO: ' + msg); process.exit(1); }
}

const prueba = `
const poi = (x, y, extra) => Object.assign(
  { x, y, cls: 'person', mature: false, mobile: false, n_obs: 12, conf: .5, dron: 1 }, extra);
const xs = () => visibles.map(p => p.x).join(', ');

console.log('1) limpiar con un solo punto dudoso B en pantalla');
estado = { pois: [poi(4, 0, { looks: 5, radius_m: 4.9, age_s: 0.5 })] };
pintar();
limpiar();
console.log('    visibles tras limpiar: [' + xs() + ']');
ok(visibles.length === 0, 'limpiar tenia que ocultar a B');

console.log('2) reporte siguiente: B sin avistamiento nuevo y un movil NUEVO a 20 m con radio 150 m');
estado = { pois: [poi(4, 0, { looks: 5, radius_m: 4.9, age_s: 2.5 }),
                  poi(20, 0, { looks: 3, radius_m: 150, mobile: true, age_s: 0.5 })] };
pintar();
console.log('    visibles: [' + xs() + ']');
ok(!visibles.some(p => p.x === 4), 'B no se volvio a ver: tiene que seguir oculto');
ok(visibles.some(p => p.x === 20), 'el movil nuevo nunca se limpio: no puede quedar oculto por su propio radio');
`;

global.ok = ok;
eval(codigo + '\n' + prueba);
console.log();
console.log('TODO OK');

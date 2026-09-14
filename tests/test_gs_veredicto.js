/*
 * Gate for the operator's verdict on the ground station page, without a browser.
 *
 * The drone cannot tell a person from an object it keeps confusing with one: measured on flight
 * 3, neither its detection rate in view, nor the detector's confidence, nor the apparent size,
 * nor an appearance classifier that transfers between flights separates them. The operator
 * looking at the crop does. So the page must (a) show how far off each point may be, and
 * (b) remember a "no es" across reports -- by position, because the list is rebuilt, reordered
 * and renumbered on every report and a POI carries no id that survives it.
 *
 * Same harness as test_gs_filtro.js: the real <script> block of gs_mapa.py, a stub DOM, and the
 * page's own pintar() / marcar() against synthetic reports.
 *
 * Run with: node tests/test_gs_veredicto.js        (from the uav_vision root)
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
  { x, y, cls: 'person', mature: true, mobile: false, n_obs: 40, conf: .6, dron: 1 }, extra);

estado = { pois: [
  poi(0, 0,  { looks: 22, radius_m: 5.2 }),
  poi(10, 0, { looks: 12, radius_m: 5.1 }),
  poi(30, 0, { mature: false, n_obs: 9 }),
]};
pintar();
const lista = () => document.getElementById('lista').innerHTML;
const cuenta = () => document.getElementById('cuenta').textContent;
console.log('  llegan 3 POI           :', cuenta());
ok(visibles.length === 3, 'sin veredictos tienen que verse los 3');
ok(lista().indexOf('&plusmn;5.2 m (95 %), 22 miradas') >= 0, 'la tarjeta no muestra el margen');
ok((lista().match(/margen/g) || []).length === 2, 'un POI sin radio no deberia inventarse un margen');
ok((lista().match(/data-v="no"/g) || []).length === 3, 'cada POI necesita su boton de descarte');

marcar(estado.pois[1], 'no');
console.log('  descarto el de x=10    :', cuenta());
ok(visibles.length === 2, 'el descartado sigue visible');
ok(cuenta().indexOf('1 descartados') >= 0, 'la cabecera tiene que decir que hay un descarte');

// A new report: the same thing moved 2 m, and the list arrives in another order. A verdict kept
// by index would now hide the wrong point and show the discarded one.
estado = { pois: [
  poi(12, 1, { looks: 14, radius_m: 5.0 }),
  poi(30, 0, { mature: false, n_obs: 11 }),
  poi(0, 0,  { looks: 25, radius_m: 5.0 }),
]};
pintar();
console.log('  reporte nuevo, reordenado:', cuenta(), '| visibles en x =', visibles.map(p => p.x).join(', '));
ok(visibles.length === 2 && !visibles.some(p => p.x === 12), 'el descarte no siguio a la cosa');
ok(visibles.some(p => p.x === 30), 'un POI a 18 m del descarte no puede ocultarse');

marcar(estado.pois[2], 'si');
console.log('  verifico el de x=0     :', (lista().match(/VERIFICADO/g) || []).length, 'chip VERIFICADO');
ok(lista().indexOf('VERIFICADO') >= 0, 'el verificado no lo dice');
ok(visibles.some(p => p.x === 0), 'verificar no puede ocultar');
`;

console.log('======================================================================');
console.log('VEREDICTO DEL OPERADOR Y MARGEN EN METROS (codigo real de gs_mapa.py)');
console.log('======================================================================');
eval(codigo + prueba);
console.log();
console.log('TODO OK');

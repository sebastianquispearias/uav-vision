/*
 * Gate for the CLIP mark on the ground station page, without a browser.
 *
 * The station scores each candidate's crop and, when the score is under the threshold, sends
 * clip_no_persona: true. The page must say so on the card, with the number, and put those cards
 * at the end of the list -- without hiding them and without touching the cards that carry no
 * score (no crop, or a station started without --clip).
 *
 * Same harness as test_gs_filtro.js: the real <script> block of gs_mapa.py, a stub DOM, and the
 * page's own pintar() against a synthetic report.
 *
 * Run with: node tests/test_gs_clip.js        (from the uav_vision root)
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
const poi = (x, extra) => Object.assign(
  { x, y: 0, cls: 'person', mature: false, mobile: false, n_obs: 9, conf: .5, dron: 1 }, extra);
const lista = () => document.getElementById('lista').innerHTML;
const marcas = () => (lista().match(/probable no persona \\(CLIP [0-9.-]+\\)/g) || []);
const tarjetas = () => lista().split('class="poi ').slice(1);

// Without scores: what a station started without --clip sends.
estado = { pois: [ poi(0, { crop: 'AAAA' }), poi(10, { crop: 'BBBB' }), poi(20) ] };
pintar();
console.log('  sin puntaje   : orden x =', visibles.map(p => p.x).join(', '), '| marcas:', marcas().length);
ok(marcas().length === 0, 'sin puntaje no puede aparecer la marca');

// The station flagged the first one. It arrives first; it must be drawn last, and marked.
estado = { pois: [
  poi(0,  { crop: 'AAAA', clip: 0.12, clip_no_persona: true }),
  poi(10, { crop: 'BBBB', clip: 2.5,  clip_no_persona: false }),
  poi(20),
]};
pintar();
console.log('  con puntaje   : orden x =', visibles.map(p => p.x).join(', '), '| marcas:', marcas().join(' / '));
ok(marcas().length === 1, 'solo el marcado lleva la marca');
ok(marcas()[0] === 'probable no persona (CLIP 0.12)', 'la marca tiene que decir el puntaje');
ok(visibles.map(p => p.x).join() === '10,20,0', 'el dudoso va al final, el resto en su orden');
ok(tarjetas()[2].indexOf('probable no persona') >= 0, 'la marca esta en la tarjeta del ultimo');
ok(tarjetas()[0].indexOf('probable no persona') < 0, 'la del confiable no la lleva');
ok((lista().match(/data-v="no"/g) || []).length === 3, 'el operador sigue pudiendo decidir sobre los 3');
ok(visibles.length === 3, 'nada se oculta');
`;

console.log('======================================================================');
console.log('MARCA CLIP Y ORDEN DE LA COLA (codigo real de gs_mapa.py)');
console.log('======================================================================');
eval(codigo + prueba);
console.log();
console.log('TODO OK');

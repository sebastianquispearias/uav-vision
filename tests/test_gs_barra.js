/*
 * Gate for the evidence bar on the ground station's candidate cards, without a browser.
 *
 * The station already printed "22 miradas" on every card, and that number cannot be read: an
 * operator who does not know that twenty is the bar has no way to tell a candidate that is almost
 * reportable from one that has barely been seen. The drone now sends the threshold and the fraction
 * next to the count, and the card draws it.
 *
 * The contrast below is the one measured on the 02ago replay against the letters a human put on
 * every box: of the four mature candidates three were nobody and one was the operator, while the
 * real people never matured. So the first card here is a ghost with a full bar and the second is a
 * real person with a quarter of one. The bar is what makes that visible, and fixing which evidence
 * fills it is separate work.
 *
 * Same harness as test_gs_capas.js: the real <script> block of gs_mapa.py, a stub DOM, and the
 * page's own pintarLista against synthetic reports.
 *
 * Run with: node tests/test_gs_barra.js        (from the uav_vision root)
 */
'use strict';
const fs = require('fs');
const path = require('path');

const GS = process.argv[2] || path.join(__dirname, '..', 'scripts', 'banco_embedded', 'gs_mapa.py');
const py = fs.readFileSync(GS, 'utf8');
const html = /PAGINA = r"""([\s\S]*?)"""/.exec(py)[1];
let codigo = /<script>([\s\S]*?)<\/script>/.exec(html)[1];
codigo = codigo.replace(/redimensionar\(\);\s*refrescar\(\);\s*setInterval\(refrescar,\s*1000\);/, '');

const nulo = new Proxy({}, { get: () => () => {}, set: () => true });
const elems = {};
function elem(id) {
  if (!elems[id]) elems[id] = {
    id, innerHTML: '', textContent: '', className: '', style: {}, dataset: {}, hidden: false,
    querySelectorAll() { return []; },
    getBoundingClientRect: () => ({ width: 800, height: 600 }),
    getContext: () => nulo,
    addEventListener() {},
    onclick: null,
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
  { x, y: 0, cls: 'person', mature: false, mobile: false, n_obs: 9, conf: .5, dron: 1,
    radius_m: 6, crop: 'AAAA' }, extra);
const lista = () => document.getElementById('lista').innerHTML;
const tarjetas = () => lista().split('class="poi ').slice(1);
const anchos = () => (lista().match(/width:([0-9]+)%/g) || []).map(s => parseInt(s.slice(6), 10));

// A ghost with its bar full, a real person at a quarter, and an old report with no fields at all.
estado = { pois: [
  poi(1, { mature: true,  looks: 20, looks_min: 20, evidence: 1 }),
  poi(2, { mature: false, looks: 5,  looks_min: 20, evidence: .25 }),
  poi(3, { mature: false, looks: null, looks_min: null, evidence: null }),
] };
pintar();

console.log('  anchos de barra        :', anchos().join('% , ') + '%');
console.log('  barras dibujadas       :', (lista().match(/class="barra"/g) || []).length, 'de 3 tarjetas');
console.log('  textos                 :', (lista().match(/[0-9]+ de [0-9]+ miradas/g) || []).join(' | '));

ok(tarjetas().length === 3, 'tienen que pintarse las tres tarjetas');
ok(anchos().length === 2, 'un reporte sin los campos no puede inventarse una barra');
ok(anchos()[0] === 100 && anchos()[1] === 25,
   'el ancho es la fraccion de evidencia, no un valor fijo ni el inverso');
ok(lista().indexOf('seen in 20 of the 20 looks it takes') >= 0 && lista().indexOf('seen in 5 of the 20 looks it takes') >= 0,
   'la barra tiene que decir el contador Y el umbral, que es el dato que faltaba');
ok(lista().indexOf('enough to report') >= 0,
   'la barra llena tiene que decir que alcanza');
ok(tarjetas()[1].indexOf('enough to report') < 0,
   'el contraste: la que va por un cuarto no puede decir que alcanza');

// The colour comes from the card's own class, so the ghost reads as confirmed and the person as
// doubtful. That is the defect, drawn: the bar is honest about what the rule believes.
ok(tarjetas()[0].slice(0, 2) === 'ok' && tarjetas()[1].slice(0, 4) === 'duda',
   'el color de la barra sale de la clase de la tarjeta');

// The margin row must keep its own wording: test_gs_veredicto.js counts the word and expects two.
ok((tarjetas()[0].match(/margin/g) || []).length === 1,
   'la barra no puede agregar otra aparicion de la palabra margin');
`;

console.log('======================================================================');
console.log('BARRA DE EVIDENCIA EN LA TARJETA (codigo real de gs_mapa.py)');
console.log('======================================================================');
eval(codigo + prueba);
console.log();
console.log('TODO OK');

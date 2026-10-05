/*
 * Gate for how far an operator's verdict reaches on the ground station page, without a browser.
 *
 * A verdict is kept by position because a POI carries no id that survives the next report. The
 * distance it is matched within used to be the larger of the verdict's radius and the other POI's
 * own 95 % radius. That radius is how far off the point may be, not how far apart two things are:
 * a mobile POI carries 68-218 m and so inherited every verdict on the map, and on the recorded
 * station a "no es" given on a doubtful point 4 m from the confirmed operator discarded the
 * operator too, while an "that is what I am looking for" marked every point OPERATOR CONFIRMED.
 *
 * Each section is a contrast: a neighbour, a mobile POI and a verified point must come out
 * differently from the POI the verdict was given on, and that POI must still carry it after it
 * moves by the report-to-report jitter of the chain.
 *
 * Same harness as test_gs_veredicto.js: the real <script> block of gs_mapa.py, a stub DOM, and the
 * page's own pintar() / marcar() against synthetic reports.
 *
 * Run with: node tests/test_gs_veredicto_radio.js        (from the uav_vision root)

  WHAT EACH SECTION PROVES

  The scene of the recorded station: the confirmed operator (A), a doubtful point 4 m from it (B)
  whose 95 % circle covers A, and a mobile POI 20 m away whose circle covers everything.
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
const cuenta = () => document.getElementById('cuenta').textContent;
const verificados = () => (document.getElementById('lista').innerHTML.match(/OPERATOR CONFIRMED/g) || []).length;
const en = x => visibles.some(p => p.x === x);
const escena = (ax, ay) => [
  poi(ax, ay, { looks: 24, radius_m: 4.9 }),
  poi(4, 0,   { looks: 21, radius_m: 4.9, clip_no_persona: true, clip: 0.31 }),
  poi(20, 0,  { looks: 20, radius_m: 150, mobile: true }),
];
const empezar = (ax, ay) => { veredictos.length = 0; estado = { pois: escena(ax, ay) }; pintar(); };

console.log('1) "no es" en B, a 4 m del operador A');
empezar(0, 0);
marcar(estado.pois[1], 'no');
console.log('   ', cuenta(), '| visibles en x =', visibles.map(p => p.x).join(', '));
ok(!en(4), 'B, sobre el que se dio el veredicto, sigue visible');
ok(en(0), 'el "no es" sobre B oculto al operador A, a 4 m');
ok(en(20), 'el "no es" sobre B oculto al POI movil de radio 150 m');

console.log('2) "no es" en A: el POI movil a 20 m no lo hereda por su propio radio');
empezar(0, 0);
marcar(estado.pois[0], 'no');
console.log('   ', cuenta(), '| visibles en x =', visibles.map(p => p.x).join(', '));
ok(!en(0), 'A, sobre el que se dio el veredicto, sigue visible');
ok(en(4), 'el "no es" sobre A oculto a B, a 4 m');
ok(en(20), 'el "no es" sobre A oculto al POI movil de radio 150 m a 20 m');

console.log('3) "that is what I am looking for" en A');
empezar(0, 0);
marcar(estado.pois[0], 'si');
console.log('   ', verificados(), 'chip OPERATOR CONFIRMED | veredictos:',
            estado.pois.map(p => p.x + '=' + veredictoDe(p)).join(', '));
ok(veredictoDe(estado.pois[0]) === 'si', 'A no quedo verificado');
ok(veredictoDe(estado.pois[1]) === null, 'verificar A marco tambien a B, a 4 m');
ok(veredictoDe(estado.pois[2]) === null, 'verificar A marco tambien al POI movil');
ok(verificados() === 1, 'tiene que haber un solo chip OPERATOR CONFIRMED');

console.log('4) el reporte siguiente trae a A corrido 1.5 m y la lista en otro orden');
empezar(0, 0);
marcar(estado.pois[0], 'no');
estado = { pois: escena(1.5, 0).reverse() };
pintar();
console.log('   ', cuenta(), '| visibles en x =', visibles.map(p => p.x).join(', '));
ok(!en(1.5), 'el "no es" no siguio a A cuando se corrio 1.5 m');
ok(en(4) && en(20), 'el "no es" de A oculto a otro POI en el reporte siguiente');
empezar(0, 0);
marcar(estado.pois[0], 'si');
estado = { pois: escena(1.5, 0).reverse() };
pintar();
console.log('    verificado corrido 1.5 m:', estado.pois.map(p => p.x + '=' + veredictoDe(p)).join(', '));
ok(veredictoDe(estado.pois[2]) === 'si', 'el "that is what I am looking for" no siguio a A corrido 1.5 m');
ok(veredictoDe(estado.pois[1]) === null && veredictoDe(estado.pois[0]) === null,
   'el "that is what I am looking for" de A paso a otro POI en el reporte siguiente');
`;

console.log('======================================================================');
console.log('ALCANCE DEL VEREDICTO DEL OPERADOR (codigo real de gs_mapa.py)');
console.log('======================================================================');
eval(codigo + prueba);
console.log();
console.log('TODO OK');

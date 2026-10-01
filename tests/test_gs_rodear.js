/*
 * The click that makes an aircraft fly, without a browser and without an aircraft.
 *
 * Every other button on this page asks the ground for something: a verdict goes to disk, a second
 * opinion goes to a detector on a laptop, a search order changes what the camera looks for. This
 * one sends a drone somewhere. So what is gated here is not that it works but that it is the only
 * one, that it goes to the point that was pressed, and that it says which aircraft went: a button
 * that says nothing when pressed reads as broken, and one that silently flies something is worse.
 *
 * Which drone goes is decided on the ground and not in the card, because the station knows who is
 * connected and the card only knows who reported the point. That half is gated in Python, in
 * tests/test_rodear.py, over the station's own dron_para_rodear.
 *
 * Same harness as test_gs_segunda.js: the real <script> block of gs_mapa.py, a stub DOM, and a
 * fetch that records instead of talking.
 *
 * Run with: node tests/test_gs_rodear.js        (from the uav_vision root)
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
    querySelectorAll(sel) {
      // Enough of a selector to find the page's own buttons in the markup it just wrote.
      const m = /^button\[data-([a-z-]+)\]$/.exec(sel);
      if (!m) return [];
      const re = new RegExp('data-' + m[1] + '="([^"]*)"', 'g');
      const out = [];
      let hit;
      while ((hit = re.exec(this.innerHTML)) !== null) {
        const ds = {};
        ds[m[1].replace(/-([a-z])/g, (s, c) => c.toUpperCase())] = hit[1];
        out.push({ dataset: ds, onclick: null });
      }
      return out;
    },
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

const pedidos = [];
global.__pedidos = pedidos;
global.fetch = (ruta, opciones) => {
  pedidos.push({ ruta, cuerpo: opciones && opciones.body ? JSON.parse(opciones.body) : null });
  // Only the flight order answers. The page polls for state on its own, and a stub that answered
  // every route with the same object would hand that poll a drone record and wipe the scene.
  if (ruta !== '/rodear') return new Promise(() => {});
  return Promise.resolve({ json: () => Promise.resolve({ dron: '2', radio_m: 30, altura_m: 25 }) });
};

function ok(cond, msg) {
  if (!cond) { console.error('FALLO: ' + msg); process.exit(1); }
}

const prueba = `
const poi = (x, extra) => Object.assign(
  { x, y: 7, cls: 'person', mature: true, mobile: false, n_obs: 40, conf: .6, dron: 1,
    radius_m: 5, crop: 'AAAA', looks: 22, looks_min: 20, evidence: 1 }, extra);
const lista = () => document.getElementById('lista').innerHTML;
const botones = (clave) =>
  document.getElementById('lista').querySelectorAll('button[data-' + clave + ']');
// The page fetches on its own (the search order, the drone list), so what is counted is orders to
// fly and not requests in general.
const rodeos = () => __pedidos.filter(q => q.ruta === '/rodear');

estado = { pois: [poi(3), poi(30, {dron: 2})] };
pintar();

console.log('  botones de rodeo       :', botones('rodear').length, 'para 2 POI');
console.log('  lo que dice el boton   :', /mirar desde otro lado/.test(lista()) ? 'mirar desde otro lado' : 'NADA');
ok(botones('rodear').length === 2, 'cada POI necesita su boton de mirar desde otro lado');
ok(lista().indexOf('mirar desde otro lado') >= 0, 'el boton tiene que decir que hace');

// Nothing flies until it is pressed. This is the assertion that matters most in this file.
console.log('  antes del clic         :', __pedidos.length, 'pedidos,', rodeos().length, 'de rodeo');
ok(rodeos().length === 0, 'pintar la lista no puede mandar a volar a nadie');

// Press the one on the second card.
pedirRodeo(estado.pois[1]);
console.log('  tras el clic           :', JSON.stringify(rodeos()[0]));
ok(rodeos().length === 1, 'un clic, una orden de vuelo');
const ult = rodeos()[0];
ok(ult.cuerpo.x === 30 && ult.cuerpo.y === 7,
   'se manda al punto de la tarjeta que se apreto, no a otro');
ok(String(ult.cuerpo.dron) === '2',
   'viaja QUIEN vio el punto, para que la estacion pueda mandar a OTRO');
ok(!('radio_m' in ult.cuerpo) && !('altura_m' in ult.cuerpo),
   'el radio y la altura los pone la estacion, que es el instrumento del operador');

// The notice names the aircraft that went, checked on the next tick because it is set when the
// answer resolves.
setTimeout(() => {
  console.log('  el aviso dice          :', aviso);
  ok(/dron 2 va a mirar/.test(aviso), 'el boton no dice que dron salio: se lee como roto');
  console.log();
  console.log('TODO OK');
}, 5);
`;

console.log('======================================================================');
console.log('MANDAR UN DRON A MIRAR DESDE OTRO LADO (codigo real de gs_mapa.py)');
console.log('======================================================================');
eval(codigo + prueba);

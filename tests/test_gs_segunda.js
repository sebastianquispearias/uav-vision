/*
 * Gate for the ground's second opinion on the station page, without a browser.
 *
 * What flies is what fits in the power budget. On the 02ago flight, over the window where the drone
 * is high, the aircraft's detector found 46.2 % of the people and RF-DETR on the laptop found
 * 90.5 % of them, with better precision. RF-DETR takes 1.4 s per frame against 35 ms, so it will
 * never fly, and this button is what lets an operator borrow it for one frame.
 *
 * Three things have to hold on the page or the button lies about what it knows:
 *
 *   1. The answer belongs to a DRONE, not to a point. The aircraft sends the frame it is looking
 *      at, which answers "is there anybody here", not "is this particular point real". A card that
 *      showed its own drone's answer as if it were about itself would promise a link between the
 *      boxes and the POI that nothing in the exchange establishes -- so two cards of the same drone
 *      show the same answer, and a card of another drone shows none.
 *   2. Every stage is visible. The exchange takes seconds: asking the drone, waiting for the frame,
 *      and the detector thinking. A button that goes quiet while that happens looks broken, and an
 *      operator who thinks it is broken clicks it again.
 *   3. A failure says so. The detector runs in another process, on another interpreter, and may not
 *      be installed at all. The map has to keep working for an operator who has no second detector.
 *
 * Same harness as test_gs_veredicto.js: the real <script> block of gs_mapa.py, a stub DOM, and the
 * page's own pintar() against synthetic reports.
 *
 * Run with: node tests/test_gs_segunda.js        (from the uav_vision root)
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
    querySelectorAll(sel) {
      // 'button[data-x]' busca por atributo; 'button' a secas devuelve los botones tal cual, que es
      // lo que la barra de busqueda usa para sus clases.
      const m0 = /\[([a-z-]+)=?/.exec(sel);
      if (!m0) {
        return [...this.innerHTML.matchAll(/<button([^>]*)>/g)]
          .map(m => ({ dataset: { b: (/data-b="([^"]+)"/.exec(m[1]) || [])[1] }, onclick: null }));
      }
      const attr = m0[1];
      // data-mirar-dron llega al dataset como mirarDron, como en un navegador de verdad.
      const clave = attr.replace(/^data-/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      return [...this.innerHTML.matchAll(new RegExp(attr + '="([^"]+)"', 'g'))]
        .map(m => ({ dataset: { v: m[1], i: m[1], mirar: m[1], [clave]: m[1] }, onclick: null }));
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

// What the page asked the station for, so the click can be checked instead of assumed. The promise
// never settles: the page's other fetches (the search order, the drone list) must not run their
// handlers here and rewrite the state this test just set up.
const pedidos = [];
global.fetch = (ruta, opciones) => {
  pedidos.push({ ruta, cuerpo: opciones && opciones.body ? JSON.parse(opciones.body) : null });
  return new Promise(() => {});
};

function ok(cond, msg) {
  if (!cond) { console.error('FALLO: ' + msg); process.exit(1); }
}

const prueba = `
const poi = (x, y, extra) => Object.assign(
  { x, y, cls: 'person', mature: true, mobile: false, n_obs: 40, conf: .6, dron: 1 }, extra);
const lista = () => document.getElementById('lista').innerHTML;
const AHORA = 1000;

// 1. before anybody asks, no card claims a second opinion
estado = { ahora: AHORA, pois: [poi(0, 0), poi(10, 0, { dron: 2 })], segunda: {} };
pintar();
console.log('  sin pedir nada         :', (lista().match(/class="segunda/g) || []).length, 'bloques de segunda opinion');
ok(lista().indexOf('class="segunda') < 0, 'invento una segunda opinion que nadie pidio');
ok((lista().match(/data-mirar=/g) || []).length === 2, 'cada POI necesita su boton de segunda opinion');

// 2. the click asks the station about the POI's DRONE
pedirSegunda(2);
console.log('  clic en el del dron 2  :', JSON.stringify(pedidos[pedidos.length - 1]));
ok(pedidos[pedidos.length - 1].ruta === '/mirar', 'el boton no pidio /mirar');
ok(pedidos[pedidos.length - 1].cuerpo.dron === '2', 'pidio la opinion sobre el dron equivocado');

// 3. every stage is visible, and none of them claims a result
for (const [fase, esperado] of [['pedido', 'asking the drone for the frame'], ['mirando', 'RF-DETR looking at the frame']]) {
  estado.segunda = { '1': { estado: fase, t: AHORA } };
  pintar();
  ok(lista().indexOf(esperado) >= 0, 'la fase ' + fase + ' no se ve en la tarjeta');
  ok(lista().indexOf('people on the ground') < 0, 'la fase ' + fase + ' ya afirma un resultado');
}
console.log('  mientras espera        : se ven las dos fases y ninguna afirma un resultado');

// 4. the answer, and to whom it belongs
estado.segunda = { '1': { estado: 'listo', t: AHORA, n: 4, espera: 1.44, dibujo: 'x.jpg',
                          personas: [{ caja: [1, 2, 3, 4], conf: .83 }] } };
pintar();
console.log('  contesta               :', /4 people on the ground, in [0-9.]+ s/.exec(lista())[0]);
ok(lista().indexOf('4 people on the ground, in 1.44 s') >= 0, 'no dice cuantas encontro ni cuanto tardo');
ok(lista().indexOf('/segunda.jpg?dron=1') >= 0, 'no muestra el cuadro que RF-DETR miro');
// The POI of drone 2 must not inherit it: the frame answers about a drone, not about a point.
const tarjetas = lista().split('<div class="poi');
ok(tarjetas.length === 3, 'se esperaban dos tarjetas, hay ' + (tarjetas.length - 1));
ok(tarjetas[1].indexOf('people on the ground') >= 0, 'la tarjeta del dron 1 perdio su respuesta');
ok(tarjetas[2].indexOf('people on the ground') < 0, 'el POI del dron 2 se apropio de la respuesta del dron 1');
console.log('  y solo al dron que miro: la tarjeta del dron 2 sigue sin respuesta');

// 5. an old answer says how old, because the scene moves
estado = Object.assign({}, estado, { ahora: AHORA + 300 });
pintar();
console.log('  cinco minutos despues  :', /\\([0-9]+ s ago\\)/.exec(lista())[0]);
ok(lista().indexOf('(300 s ago)') >= 0, 'una respuesta vieja no dice que lo es');

// 6. a drone with no points still has a button: it has a camera, and it is exactly the drone an
// operator wonders about. With the button only inside a point's card, the second opinion was
// unreachable for a drone that reports nothing -- which is when you most want to look.
estado = { ahora: AHORA, pois: [], segunda: {},
           drones: {'1': {buscando: {clases: ['person'], v: null, epoca: null}},
                    '7': {buscando: {clases: ['person'], v: null, epoca: null}}} };
ultimoEstado = estado;
pintarBuscar();
const barra = document.getElementById('buscar').innerHTML;
console.log('  sin un solo punto en el mapa:',
            (barra.match(/data-mirar-dron/g) || []).length, 'botones de segunda opinion');
ok((barra.match(/data-mirar-dron/g) || []).length === 2,
   'cada dron conectado necesita su boton, aunque no haya reportado ni un punto');
ok(barra.indexOf('data-mirar-dron="7"') >= 0, 'falta el boton del dron 7');
pedirSegunda('7');
ok(pedidos[pedidos.length - 1].cuerpo.dron === '7',
   'el boton de un dron tiene que preguntar por ESE dron');

// 7. no detector installed: it says so instead of going quiet
estado = { ahora: AHORA + 300, pois: [poi(0, 0), poi(10, 0, { dron: 2 })],
           segunda: { '1': { estado: 'error', t: AHORA + 300, error: 'no existe el venv' } } };
pintar();
console.log('  sin detector en tierra :', /no second opinion: [^<]*/.exec(lista())[0]);
ok(lista().indexOf('no second opinion: no existe el venv') >= 0, 'un fallo no dice por que');
ok(lista().indexOf('/segunda.jpg') < 0, 'muestra una imagen que no existe');
ok(visibles.length === 2, 'un fallo de la segunda opinion no puede tocar el mapa');
`;

console.log('======================================================================');
console.log('SEGUNDA OPINION EN LA ESTACION (codigo real de gs_mapa.py)');
console.log('======================================================================');
eval(codigo + prueba);
console.log();
console.log('TODO OK');

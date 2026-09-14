// What the station offers to search for, and what it says each drone is doing.
//
// The buttons used to be a fixed list, 'boat' included, on a drone whose model has no boat. They
// now come from the classes the drones report their detector can emit, with the model's names for
// people collapsed into the one an operator types. And under them, one line per drone with its
// applied state, because the station having recorded a request is no evidence a drone took it.
//
// Runs the real page code of gs_mapa.py with no browser, like test_gs_filtro.js.
//
// Run: node tests/test_gs_buscar.js
const fs = require('fs');
const path = require('path');

const GS = process.argv[2] || path.join(__dirname, '..', 'scripts', 'banco_embedded', 'gs_mapa.py');
const py = fs.readFileSync(GS, 'utf8');
const html = /PAGINA = r"""([\s\S]*?)"""/.exec(py)[1];
const codigo = /<script>([\s\S]*?)<\/script>/.exec(html)[1];

const nulo = new Proxy(function () {}, { get: () => nulo, apply: () => nulo });
const elems = {};
function elem(id) {
  if (!elems[id]) elems[id] = {
    id, innerHTML: '', textContent: '', className: '', style: {}, dataset: {},
    querySelectorAll: () => [],
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

// The checks share the page's scope, so their names carry a suffix that no page global uses.
const prueba = `
ok(buscables({}).join(',') === BUSCABLES.join(','),
   'sin drones que informen, los botones tienen que ser la lista fija');
console.log('  sin informes      :', buscables({}).join(', '));

const dronesPrueba = {
  '1': { buscando: { clases: ['car'], v: 3, epoca: 'e1', rechazo: null,
                     conocidas: ['bus', 'car', 'pedestrian', 'people', 'truck'] } },
  '2': { buscando: { clases: null, v: 2, epoca: 'e1', rechazo: null,
                     conocidas: ['boat', 'car', 'person'] } },
  '3': { t: 0 },
};
const botonesPrueba = buscables(dronesPrueba);
console.log('  con informes      :', botonesPrueba.join(', '));
ok(botonesPrueba.join(',') === 'person,boat,bus,car,truck',
   'los botones tienen que ser la union de lo que los modelos emiten, persona primero');
ok(!botonesPrueba.includes('pedestrian') && !botonesPrueba.includes('people'),
   'los nombres de persona de un modelo no pueden salir como botones sueltos');

const ordenPrueba = { v: 3, epoca: 'e1' };
const lineasPrueba = estadoBusqueda(dronesPrueba, ordenPrueba);
lineasPrueba.forEach(l => console.log('    ' + l));
ok(lineasPrueba[0] === 'dron 1: busca car', 'un dron al dia tiene que decir que busca');
ok(lineasPrueba[1] === 'dron 2: todavia no tomo la orden', 'un dron atrasado no puede verse al dia');
ok(lineasPrueba[2] === 'dron 3: no informa que busca', 'un dron sin estado no puede inventarse uno');

dronesPrueba['1'].buscando.rechazo = "the detector does not emit ['boat']";
ok(estadoBusqueda(dronesPrueba, ordenPrueba)[0].startsWith('dron 1: rechazo la orden'),
   'un rechazo tiene que verse como rechazo');
dronesPrueba['1'].buscando.rechazo = null;

ok(estadoBusqueda(dronesPrueba, { v: 3, epoca: 'e9' })[0] === 'dron 1: todavia no tomo la orden',
   'la misma version de otra sesion de la estacion no es la misma orden');
ok(estadoBusqueda(dronesPrueba, { v: 0, epoca: 'e1' })[1] === 'dron 2: busca lo de siempre',
   'sin ninguna orden dada, nadie esta atrasado');
console.log('  rechazo, sesion distinta y ninguna orden: correctos');
`;

console.log('======================================================================');
console.log('BUSCAR DE LA GROUND STATION (codigo real de gs_mapa.py)');
console.log('======================================================================');
eval(codigo + prueba);
console.log();
console.log('TODO OK');

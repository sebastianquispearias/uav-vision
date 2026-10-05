/*
  What the station offers to search for, and what it says each drone is doing.

  The buttons used to be a fixed list, 'boat' included, on a drone whose model has no boat. They
  now come from the classes the drones report their detector can emit, with the model's names for
  people collapsed into the one an operator types. And under them, one line per drone with its
  applied state, because the station having recorded a request is no evidence a drone took it.

  Runs the real page code of gs_mapa.py with no browser, like test_gs_filtro.js.

  Run: node tests/test_gs_buscar.js

  WHAT EACH SECTION PROVES

  The checks share the page's scope, so their names carry a suffix that no page global uses.

  Un detector entrenado con un conjunto publico conoce ochenta cosas, y la mayoria no tiene nada
  que hacer en un mapa de busqueda. Ofrecerlas todas entierra los cuatro botones que un operador
  va a apretar. Pero quitarlas seria mentir sobre lo que el detector puede emitir, asi que quedan
  a un clic. Esta seccion fija las dos mitades de esa decision.
*/
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

const prueba = `
ok(buscables({}).mision.join(',') === BUSCABLES.join(','),
   'sin drones que informen, los botones tienen que ser la lista fija');
console.log('  sin informes      :', buscables({}).mision.join(', '));

const dronesPrueba = {
  '1': { buscando: { clases: ['car'], v: 3, epoca: 'e1', rechazo: null,
                     conocidas: ['bus', 'car', 'pedestrian', 'people', 'truck'] } },
  '2': { buscando: { clases: null, v: 2, epoca: 'e1', rechazo: null,
                     conocidas: ['boat', 'car', 'person'] } },
  '3': { t: 0 },
};
const botonesPrueba = buscables(dronesPrueba).mision;
console.log('  con informes      :', botonesPrueba.join(', '));
ok(botonesPrueba.join(',') === 'person,car,truck,bus,boat',
   'los botones tienen que ser las clases de la mision que los modelos emiten, en ese orden');
ok(!botonesPrueba.includes('pedestrian') && !botonesPrueba.includes('people'),
   'los nombres de persona de un modelo no pueden salir como botones sueltos');

const cocoPrueba = {
  '1': { buscando: { conocidas: ['person', 'car', 'broccoli', 'teddy bear', 'banana', 'boat'] } },
};
const cortadoPrueba = buscables(cocoPrueba);
console.log('  con un modelo COCO:', cortadoPrueba.mision.join(', '),
            '| escondidas:', cortadoPrueba.resto.join(', '));
ok(cortadoPrueba.mision.join(',') === 'person,car,boat',
   'solo pueden ofrecerse las clases de la mision que el detector emite');
ok(!cortadoPrueba.mision.includes('broccoli'),
   'el brocoli no puede estar entre los botones de una estacion de busqueda');
ok(cortadoPrueba.resto.join(',') === 'banana,broccoli,teddy bear',
   'las demas no se borran: quedan disponibles, porque el detector si puede emitirlas');

const ordenPrueba = { v: 3, epoca: 'e1' };
const lineasPrueba = estadoBusqueda(dronesPrueba, ordenPrueba);
lineasPrueba.forEach(l => console.log('    ' + l));
ok(lineasPrueba[0] === 'drone 1: looking for car', 'un dron al dia tiene que decir que busca');
ok(lineasPrueba[1] === 'drone 2: has not taken the order yet', 'un dron atrasado no puede verse al dia');
ok(lineasPrueba[2] === 'drone 3: does not say what it is looking for', 'un dron sin estado no puede inventarse uno');

dronesPrueba['1'].buscando.rechazo = "the detector does not emit ['boat']";
ok(estadoBusqueda(dronesPrueba, ordenPrueba)[0].startsWith('drone 1: refused the order'),
   'un rechazo tiene que verse como rechazo');
dronesPrueba['1'].buscando.rechazo = null;

ok(estadoBusqueda(dronesPrueba, { v: 3, epoca: 'e9' })[0] === 'drone 1: has not taken the order yet',
   'la misma version de otra sesion de la estacion no es la misma orden');
ok(estadoBusqueda(dronesPrueba, { v: 0, epoca: 'e1' })[1] === 'drone 2: looking for the usual',
   'sin ninguna orden dada, nadie esta atrasado');
console.log('  rechazo, sesion distinta y ninguna orden: correctos');
`;

console.log('======================================================================');
console.log('BUSCAR DE LA GROUND STATION (codigo real de gs_mapa.py)');
console.log('======================================================================');
eval(codigo + prueba);
console.log();
console.log('TODO OK');

/*
 * Gate for the health tab on the ground station, without a browser.
 *
 * The tab exists because of a question nobody on this screen could answer. On 2026-10-02 a
 * visitor read "4 FPS" off the header and asked why a Pi 4 and a Pi 5 were reporting the same
 * number. They were not: the header was showing the rate both boards had been ASKED for, and
 * the Pi 4 was delivering 0.78 of 3.00 while throttling at 84 C. Every piece of that was
 * invisible, and the header still only shows the FIRST drone, so with two aircraft the second
 * one has no rate on the page at all.
 *
 * The rows are ordered by HOW SOON EACH THING ENDS THE FLIGHT, which is the rule the alarm
 * banner already follows. That ordering is the design, so it is asserted here: a test that only
 * checked the numbers were present would pass on a table sorted alphabetically.
 *
 * Same harness as test_gs_barra.js: the real <script> block of gs_mapa.py, a stub DOM, and the
 * page's own pintarDiagnostico against synthetic reports.
 *
 * Run with: node tests/test_gs_diagnostico.js        (from the uav_vision root)

  WHAT EACH SECTION PROVES

  1. The two boards of the real bench, side by side, each cell coloured by its own threshold.
     The contrast is the whole point: the same table, one board fine and one board dying, and
     the difference legible without reading a number twice.
  2. The order of the rows is the order of urgency, not any order that happens to come out.
  3. Volts per cell and not percent, and the cell count inferred rather than assumed: a 4S and
     a 6S pack at the same pack voltage are not in the same state.
  4. A field that did not arrive says so instead of reading as a zero. This is the one that
     matters most: a dash is honest, a 0.0 V/cell on a healthy aircraft is a false alarm and a
     0 missed slots on a board that never reported is a lie.
  5. The sticky bit. A board that browned out an hour ago still says so.
  6. The tab carries a count of what is red, because the operator is looking at the map.
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
// The tab strip is the one place this page uses querySelectorAll on the document, so the stub
// has to answer it with something clickable or the switcher silently does nothing.
const solapas = [
  Object.assign(elem('solapa-contactos'), { dataset: { solapa: 'contactos' }, _oyentes: [] }),
  Object.assign(elem('solapa-diag'), { dataset: { solapa: 'diag' }, _oyentes: [] }),
];
// addEventListener('click', fn): the handler is the SECOND argument. A stub that kept the first
// one stored the string 'click' and the click did nothing, which is also how the real page would
// fail if the switcher were wired to the wrong argument.
solapas.forEach(b => { b.addEventListener = (ev, fn) => b._oyentes.push(fn); });
// The page asks the CONTAINER for its buttons, so that is what the stub has to answer.
elem('solapas').querySelectorAll = sel => (sel === 'button' ? solapas : []);
global.document = { getElementById: elem };
global.window = { devicePixelRatio: 1, addEventListener: () => {} };
global.setInterval = () => 0;
global.fetch = () => new Promise(() => {});

function ok(cond, msg) {
  if (!cond) { console.error('FALLO: ' + msg); process.exit(1); }
  console.log('  [ok  ] ' + msg);
}

const prueba = `
const tabla = () => document.getElementById('diag').innerHTML;
// The cells of one row, in column order, stripped of their markup.
function fila(etiqueta) {
  const t = tabla();
  const i = t.indexOf('<th>' + etiqueta + '</th>');
  if (i < 0) return null;
  const corte = t.slice(i, t.indexOf('</tr>', i));
  return (corte.match(/<td[^>]*>[\\s\\S]*?<\\/td>/g) || []).map(c => ({
    clase: (/class="([^"]*)"/.exec(c) || [, ''])[1],
    texto: c.replace(/<[^>]*>/g, ' ').replace(/\\s+/g, ' ').trim(),
  }));
}
function ordenFilas() {
  return (tabla().match(/<th>[A-Z][^<]*<\\/th>/g) || []).map(s => s.slice(4, -5));
}

console.log('======================================================================');
console.log('1. LAS DOS PLACAS DEL BANCO, UNA SANA Y UNA MURIENDOSE');
console.log('======================================================================');

// The numbers are the ones measured on this bench: the Pi 5 passed its load test at 15.0 V with
// throttled=0x0, and the Pi 4 gave 0.78 FPS of the 3.00 the mission asks for, at 84 C.
estado = { drones: {
  '1': { bateria_v: 15.013, temp_c: 48.3, fps_real: 2.95, fps_pedido: 3.0,
         slots_perdidos: 0, slots_perdidos_total: 0,
         salud: { crudo: 0, ahora: [], alguna_vez: [] } },
  '2': { bateria_v: 13.1, temp_c: 84.0, fps_real: 0.78, fps_pedido: 3.0,
         slots_perdidos: 3, slots_perdidos_total: 41,
         salud: { crudo: 0x80009, ahora: ['bajo_voltaje', 'limite_termico'],
                  alguna_vez: ['limite_termico'] } },
} };
pintarDiagnostico();

['Cell voltage', 'Supply dips', 'Temperature', 'Rate delivered', 'Slots missed'].forEach(n => {
  const f = fila(n);
  console.log('  ' + (n + '              ').slice(0, 15) + ': '
    + f.map(c => '[' + c.clase.replace('dato ', '') + '] ' + c.texto).join('   |   '));
});

const v = fila('Cell voltage');
ok(v.length === 2, 'hay una columna por dron, no solo la del primero');
ok(v[0].clase.indexOf('bien') >= 0, 'la Pi 5 a 15,013 V sale en verde');
ok(v[1].clase.indexOf('mal') >= 0 && v[1].texto.indexOf('land now') >= 0,
   'la Pi 2 a 13,1 V (3,28 V/celda) dice ATERRIZAR, no solo un color');

const t = fila('Rate delivered');
ok(t[0].texto.indexOf('2.95 of 3 FPS') >= 0,
   'la tasa se muestra CONTRA la pedida, que es lo que faltaba en la cabecera');
ok(t[0].clase.indexOf('bien') >= 0 && t[1].clase.indexOf('mal') >= 0,
   '2,95 de 3 es verde y 0,78 de 3 es rojo: el mismo numero de FPS no es el mismo estado');
ok(t[1].texto.indexOf('26 %') >= 0, 'y dice la fraccion: 0,78 de 3,00 es el 26 %');

const c = fila('Temperature');
ok(c[0].clase.indexOf('bien') >= 0 && c[1].clase.indexOf('mal') >= 0,
   '48,3 C en verde y 84,0 C en rojo');
ok(c[1].texto.indexOf('throttling') >= 0,
   'y cuando el bit termico esta puesto AHORA lo DICE, porque es lo que cuesta miradas');
ok(c[0].texto.indexOf('throttling') < 0,
   'la placa fria no lo dice: el contraste es lo que hace legible la de al lado');

// El contraste que importa en esta fila: el bit termico NO es pegajoso aqui a proposito. Una
// placa que freno bajo carga hace una hora y ahora esta a 50 C puede volar; la de abajo, no. En
// la fila de la tension es al reves, y la seccion 5 lo prueba.
estado = { drones: { '1': { temp_c: 50.2,
                            salud: { crudo: 0x80000, ahora: [], alguna_vez: ['limite_termico'] } } } };
pintarDiagnostico();
const frio = fila('Temperature')[0];
console.log('  freno en el pasado, hoy a 50,2 C: [' + frio.clase.replace('dato ', '') + '] ' + frio.texto);
ok(frio.texto.indexOf('throttling') < 0 && frio.clase.indexOf('bien') >= 0,
   'una placa que freno ANTES y hoy esta fria no arrastra el aviso: puede volar');

console.log();
console.log('======================================================================');
console.log('2. EL ORDEN ES EL DE LA URGENCIA, Y EL ORDEN ES EL DISENO');
console.log('======================================================================');
console.log('  filas, de arriba a abajo:', ordenFilas().join(' -> '));
ok(JSON.stringify(ordenFilas())
   === JSON.stringify(['Cell voltage', 'Supply dips', 'Temperature',
                       'Rate delivered', 'Slots missed']),
   'minutos, segundos, degrada ya, degrada el producto, idem: en ese orden');

console.log();
console.log('======================================================================');
console.log('3. VOLTIOS POR CELDA, Y LAS CELDAS SE INFIEREN');
console.log('======================================================================');
[[15.013, 4, 3.75], [22.2, 6, 3.70], [11.1, 3, 3.70], [7.4, 2, 3.70]].forEach(([pack, n]) => {
  console.log('  ' + pack + ' V -> ' + celdasDe(pack) + 'S -> '
    + (pack / celdasDe(pack)).toFixed(2) + ' V/celda');
  ok(celdasDe(pack) === n, pack + ' V se reconoce como ' + n + 'S');
});
console.log('  5.0 V (un cargador USB, no un pack) ->', celdasDe(5.0));
ok(celdasDe(5.0) === null, 'una fuente que no es LiPo devuelve null en vez de inventar celdas');

// LA COTA ALTA ES LA QUE SOSTIENE ESTA FILA, y una mas suelta estaba MAL. A 4,35 V por celda,
// 13,0 V salian como 3S a 4,33 -> verde, pack lleno. Los mismos 13,0 V sobre 4S son 3,25 ->
// rojo, aterrizar. Un numero, dos lecturas, y la equivocada es la tranquilizadora.
console.log('  13.0 V ->', celdasDe(13.0) + 'S -> ' + (13.0 / celdasDe(13.0)).toFixed(2) + ' V/celda');
ok(celdasDe(13.0) === 4,
   '13,0 V son un 4S agotado y no un 3S lleno: 4,33 V/celda no existe en una LiPo');

console.log();
console.log('  y el caso que la inferencia NO puede resolver, que es por lo que hay --celdas:');
console.log('    12.0 V inferidos ->', celdasDe(12.0) + 'S a '
  + (12.0 / celdasDe(12.0)).toFixed(2) + ' V/celda');
console.log('    12.0 V dichos 4S ->', celdasDe(12.0, 4) + 'S a '
  + (12.0 / celdasDe(12.0, 4)).toFixed(2) + ' V/celda');
ok(celdasDe(12.0) === 3 && celdasDe(12.0, 4) === 4,
   'un 3S cargado a 4,00 y un 4S agotado a 3,00 dan los mismos 12,0 V');

estado = { celdas: 4, drones: { '1': { bateria_v: 12.0 } } };
pintarDiagnostico();
const dicho = fila('Cell voltage')[0];
console.log('    con --celdas 4 la pestana dice: [' + dicho.clase.replace('dato ', '') + '] ' + dicho.texto);
ok(dicho.texto.indexOf('3.00') >= 0 && dicho.clase.indexOf('mal') >= 0,
   '--celdas 4 convierte esos 12,0 V en 3,00 V/celda y en ATERRIZAR');
ok(dicho.texto.indexOf('guessed') < 0, 'y deja de decir que lo adivino');

estado = { drones: { '1': { bateria_v: 12.0 } } };
pintarDiagnostico();
const adivinado = fila('Cell voltage')[0];
console.log('    sin --celdas la pestana dice : [' + adivinado.clase.replace('dato ', '') + '] ' + adivinado.texto);
ok(adivinado.texto.indexOf('guessed') >= 0,
   'sin la bandera AVISA de que el numero de celdas es una conjetura, en vez de callarselo');

estado = { drones: { '1': { bateria_v: 22.2 }, '2': { bateria_v: 14.8 } } };
pintarDiagnostico();
const pc = fila('Cell voltage');
console.log('  22.2 V y 14.8 V ->', pc.map(x => x.texto).join('   |   '));
ok(pc[0].texto.indexOf('3.70') >= 0 && pc[0].texto.indexOf('6S') >= 0,
   '22,2 V se leen como 3,70 V/celda sobre 6S');
ok(pc[1].texto.indexOf('3.70') >= 0 && pc[1].texto.indexOf('4S') >= 0,
   'y 14,8 V como 3,70 V/celda sobre 4S: packs distintos, MISMO estado');
ok(pc[0].clase === pc[1].clase,
   'por eso los dos salen del mismo color, que es lo que el porcentaje no dice');

console.log();
console.log('  EL BORDE EXACTO, que en el banco se vio porque el pack falso se para ahi:');
estado = { celdas: 4, drones: {
  '1': { bateria_v: 14.00 },   // 3.50 clavados: el umbral de "turn back"
  '2': { bateria_v: 13.20 },   // 3.30 clavados: el umbral de "land now"
  '3': { bateria_v: 14.04 },   // 3.51, apenas por encima
} };
pintarDiagnostico();
const borde = fila('Cell voltage');
borde.forEach((c2, i) => console.log('    ' + ['14.00', '13.20', '14.04'][i]
  + ' V -> [' + c2.clase.replace('dato ', '') + '] ' + c2.texto));
ok(borde[0].clase.indexOf('ojo') >= 0 && borde[0].texto.indexOf('turn back') >= 0,
   '3,50 CLAVADOS ya dice turn back: el pie dice 3,50 y la celda no puede contradecirlo');
ok(borde[1].clase.indexOf('mal') >= 0 && borde[1].texto.indexOf('land now') >= 0,
   '3,30 clavados ya dice land now');
ok(borde[2].clase.indexOf('bien') >= 0,
   '3,51 sigue en verde: el umbral mueve el color en el borde, no antes');

console.log();
console.log('======================================================================');
console.log('4. LO QUE NO LLEGO SE DICE, NO SE PINTA COMO UN CERO');
console.log('======================================================================');
estado = { drones: { '9': { fps_real: 1.9 } } };
pintarDiagnostico();
['Cell voltage', 'Supply dips', 'Temperature', 'Slots missed'].forEach(n => {
  const f = fila(n);
  console.log('  ' + (n + '              ').slice(0, 15) + ': [' + f[0].clase + '] ' + f[0].texto);
  ok(f[0].clase === 'nada', n + ': un campo ausente sale como "sin dato"');
  ok(!/^[0-9]/.test(f[0].texto), n + ': y no empieza por un numero inventado');
});
const tr = fila('Rate delivered');
console.log('  Rate delivered : [' + tr[0].clase + '] ' + tr[0].texto);
ok(tr[0].texto.indexOf('1.9 FPS') >= 0 && tr[0].texto.indexOf('asked rate unknown') >= 0,
   'una tasa sin la pedida se muestra, pero diciendo que falta la referencia');
ok(tr[0].clase.indexOf('mal') < 0 && tr[0].clase.indexOf('bien') < 0,
   'y sin color, porque sin referencia no hay juicio que dar');

estado = { drones: {} };
pintarDiagnostico();
console.log('  sin ningun dron:', tabla().replace(/<[^>]*>/g, '').trim());
ok(tabla().indexOf('No drone has reported yet') >= 0,
   'sin drones la pestana lo dice y no muestra una tabla vacia');

console.log();
console.log('======================================================================');
console.log('5. EL BIT PEGAJOSO: UNA CAIDA DE HACE UNA HORA SIGUE CONTANDO');
console.log('======================================================================');
estado = { drones: {
  '1': { salud: { crudo: 0x10000, ahora: [], alguna_vez: ['bajo_voltaje'] } },
  '2': { salud: { crudo: 0x1, ahora: ['bajo_voltaje'], alguna_vez: ['bajo_voltaje'] } },
  '3': { salud: { crudo: 0, ahora: [], alguna_vez: [] } },
} };
pintarDiagnostico();
const s = fila('Supply dips');
console.log('  pasado / ahora / limpio:', s.map(x => '[' + x.clase.replace('dato ', '') + '] ' + x.texto).join('   |   '));
ok(s[0].texto.indexOf('has browned out') >= 0 && s[0].clase.indexOf('ojo') >= 0,
   'un bit de "alguna vez" con el de "ahora" apagado SIGUE avisando');
ok(s[1].texto.indexOf('NOW') >= 0 && s[1].clase.indexOf('mal') >= 0,
   'y uno que esta cayendo ahora mismo se distingue del anterior');
ok(s[2].clase.indexOf('bien') >= 0, 'una placa limpia dice que esta limpia');

console.log();
console.log('======================================================================');
console.log('6. LA SOLAPA LLAMA SOLA CUANDO HAY ALGO EN ROJO');
console.log('======================================================================');
const botones = botonesSolapa();
const laDeDiag = botones.filter(b => b.dataset.solapa === 'diag')[0];

estado = { drones: { '1': { bateria_v: 15.013, temp_c: 48.3, fps_real: 3.0, fps_pedido: 3.0,
                            slots_perdidos: 0,
                            salud: { crudo: 0, ahora: [], alguna_vez: [] } } } };
pintarDiagnostico(); pintarSolapas();
console.log('  flota sana   -> la solapa dice:', JSON.stringify(laDeDiag.innerHTML));
ok(laDeDiag.innerHTML.indexOf('\\u25cf') < 0, 'con todo en verde la solapa no grita');

estado = { drones: { '1': { bateria_v: 13.0, temp_c: 84.0, fps_real: 0.78, fps_pedido: 3.0,
                            slots_perdidos: 3,
                            salud: { crudo: 0x1, ahora: ['bajo_voltaje'], alguna_vez: ['bajo_voltaje'] } } } };
pintarDiagnostico(); pintarSolapas();
console.log('  flota muriendo -> la solapa dice:', JSON.stringify(laDeDiag.innerHTML));
ok(laDeDiag.innerHTML.indexOf('\\u25cf') >= 0,
   'con celdas en rojo la solapa lleva la cuenta, porque el operador mira el mapa');
ok(/\\u25cf 4/.test(laDeDiag.innerHTML),
   'y la cuenta es la de celdas rojas: tension, caida, temperatura y tasa');

console.log();
console.log('======================================================================');
console.log('7. CAMBIAR DE PESTANA MUESTRA UNA Y ESCONDE LA OTRA');
console.log('======================================================================');
pintarSolapas();
console.log('  al abrir   -> contactos visible:', !document.getElementById('panel-contactos').hidden,
            ', diagnostico visible:', !document.getElementById('panel-diag').hidden);
ok(!document.getElementById('panel-contactos').hidden
   && document.getElementById('panel-diag').hidden,
   'la pagina abre en contactos: el mapa es el producto');

laDeDiag._oyentes.forEach(fn => fn());
console.log('  tras clic  -> contactos visible:', !document.getElementById('panel-contactos').hidden,
            ', diagnostico visible:', !document.getElementById('panel-diag').hidden);
ok(document.getElementById('panel-contactos').hidden
   && !document.getElementById('panel-diag').hidden,
   'el clic en Health muestra el diagnostico Y esconde los contactos');
ok(laDeDiag.className === 'on', 'y la solapa activa se marca');
`;

eval(codigo + prueba);
console.log();
console.log('TODO OK');

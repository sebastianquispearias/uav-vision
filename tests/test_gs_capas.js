/*
 * Gate for the two layers of the ground station map, without a browser.
 *
 * Tactical displays treat a track that has stopped being refreshed as a claim that is losing its
 * value: it fades and then leaves the screen. Before this gate the page drew every POI it had ever
 * received, so a person lost a minute ago kept its pin, its "95 %" circle of more than a hundred
 * metres and its card, and the map filled with things that were no longer true.
 *
 * Two layers, then. LIVE contacts (not confirmed, not verified by the operator) fade with age_s
 * and are hidden past the cap. PERSISTENT ones (mature, or marked "es lo que busco") stay at their
 * last position and say how long ago they were seen. "limpiar" hides the live contacts on screen
 * until a newer sighting brings them back; "historial" shows everything the layers hid.
 *
 * Same harness as test_gs_veredicto.js: the real <script> block of gs_mapa.py, a stub DOM, and the
 * page's own functions against synthetic reports. The canvas context records the alpha each
 * stroke is drawn with, so the 95 % circle fading with its pin is observed, not assumed.
 *
 * Run with: node tests/test_gs_capas.js        (from the uav_vision root)
 */
'use strict';
const fs = require('fs');
const path = require('path');

const GS = process.argv[2] || path.join(__dirname, '..', 'scripts', 'banco_embedded', 'gs_mapa.py');
const py = fs.readFileSync(GS, 'utf8');
const html = /PAGINA = r"""([\s\S]*?)"""/.exec(py)[1];
let codigo = /<script>([\s\S]*?)<\/script>/.exec(html)[1];
codigo = codigo.replace(/redimensionar\(\);\s*refrescar\(\);\s*setInterval\(refrescar,\s*1000\);/, '');

// A 2D context that remembers, for every dashed stroke, the alpha it was drawn with.
const trazos = [];
const lienzo2d = new Proxy({ globalAlpha: 1, _dash: [] }, {
  get(o, k) {
    if (k in o) return o[k];
    return (...args) => {
      if (k === 'setLineDash') o._dash = args[0] || [];
      if (k === 'stroke' && o._dash.length) trazos.push(o.globalAlpha);
    };
  },
  set(o, k, v) { o[k] = v; return true; },
});
const elems = {};
function elem(id) {
  if (!elems[id]) elems[id] = {
    id, innerHTML: '', textContent: '', className: '', style: {}, dataset: {}, hidden: false,
    querySelectorAll() { return []; },
    getBoundingClientRect: () => ({ width: 800, height: 600 }),
    getContext: () => lienzo2d,
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
  { x, y: 0, cls: 'person', mature: false, mobile: true, n_obs: 9, conf: .5, dron: 1,
    radius_m: 6, crop: 'AAAA' }, extra);
const lista = () => document.getElementById('lista').innerHTML;
const tarjetas = () => lista().split('class="poi ').slice(1);
const xs = () => visibles.map(p => p.x).join(',');
console.log('  tope de contacto vivo: ' + TOPE_VIVO_S + ' s');
ok(typeof TOPE_VIVO_S === 'number' && TOPE_VIVO_S > 3, 'falta un tope mayor que la extrapolacion');

// 1. Fresh, old-but-inside-the-cap, beyond the cap, and mature beyond the cap.
const viejo = TOPE_VIVO_S + 55;
estado = { pois: [
  poi(0,  { age_s: 0 }),
  poi(10, { age_s: TOPE_VIVO_S - 1 }),
  poi(20, { age_s: viejo }),
  poi(30, { age_s: viejo, mature: true }),
]};
trazos.length = 0;
pintar();
console.log('  llegan edades 0 / tope-1 / ' + viejo + ' / ' + viejo + ' maduro -> visibles x =', xs(),
            '| alfa', visibles.map(p => alfaDe(p).toFixed(2)).join(' '));
ok(xs() === '0,10,30', 'el contacto viejo no maduro tiene que desaparecer y el maduro quedarse');
ok(alfaDe(estado.pois[0]) === 1, 'un POI recien visto va a opacidad plena');
ok(alfaDe(estado.pois[1]) > 0 && alfaDe(estado.pois[1]) < 0.6, 'cerca del tope tiene que estar desvanecido');
ok(alfaDe(estado.pois[3]) === 1, 'un persistente no se desvanece');
ok(tarjetas().length === 3, 'la lista tiene que seguir al mapa: ' + tarjetas().length + ' tarjetas');
ok(lista().indexOf('x=20') < 0 && !tarjetas().some(t => t.indexOf('20 m E') >= 0), 'el oculto sigue en la lista');
const hace = tarjetas().filter(t => /visto hace \\d+ s/.test(t));
console.log('  tarjetas con "visto hace":', hace.map(t => /visto hace \\d+ s/.exec(t)[0]).join(' | '));
ok(hace.length === 1 && hace[0].indexOf('30 m E') >= 0, 'solo el persistente pasado el tope dice visto hace');
ok(hace[0].indexOf('visto hace ' + viejo + ' s') >= 0, 'visto hace tiene que dar la edad');
console.log('  alfa de los circulos 95 % dibujados:', trazos.map(a => a.toFixed(2)).join(' '));
ok(trazos.length === 3, 'el circulo de un POI oculto no puede dibujarse');
ok(trazos.some(a => a > 0 && a < 0.6), 'el circulo del desvanecido tiene que desvanecerse con el');
ok(trazos.filter(a => a === 1).length === 2, 'el fresco y el persistente dibujan su circulo pleno');

// 2. Marked "es lo que busco": persistent regardless of age.
estado = { pois: [ poi(0, { age_s: viejo }) ]};
pintar();
ok(visibles.length === 0, 'antes del veredicto un viejo no maduro no se ve');
marcar(poi(0, { age_s: 0 }), 'si');
pintar();
console.log('  viejo marcado "es lo que busco" -> visibles', visibles.length, '| alfa', alfaDe(estado.pois[0]));
ok(visibles.length === 1 && alfaDe(estado.pois[0]) === 1, 'lo que el operador verifico no puede desaparecer');
ok(/visto hace \\d+ s/.test(lista()), 'el verificado viejo tiene que decir hace cuanto se vio');

// 3. limpiar: hides live contacts on screen until a newer sighting.
estado = { pois: [ poi(50, { age_s: 1 }), poi(70, { age_s: 1, mature: true }) ]};
pintar();
ok(xs() === '50,70', 'antes de limpiar se ven los dos');
limpiar();
console.log('  limpiar -> visibles x =', xs());
ok(xs() === '70', 'limpiar tiene que ocultar el vivo y respetar el confirmado');
estado = { pois: [ poi(50, { age_s: 3 }), poi(70, { age_s: 3, mature: true }) ]};
pintar();
console.log('  reporte sin avistamiento nuevo (age_s 1 -> 3) -> visibles x =', xs());
ok(xs() === '70', 'sin avistamiento nuevo el limpiado no vuelve');
estado = { pois: [ poi(50.5, { age_s: 0.2 }), poi(70, { age_s: 5, mature: true }) ]};
pintar();
console.log('  reporte con avistamiento nuevo (age_s 0.2) -> visibles x =', xs());
ok(xs() === '50.5,70', 'un avistamiento nuevo tiene que traerlo de vuelta');
estado = { pois: [ poi(50.5, { age_s: 2 }), poi(70, { age_s: 7, mature: true }) ]};
pintar();
ok(xs() === '50.5,70', 'una vez de vuelta, envejecer dentro del tope no lo vuelve a ocultar');

// 4. historial: everything the layers hid, faded; the verdict "no es" is not a layer.
estado = { pois: [ poi(90, { age_s: 1 }), poi(100, { age_s: viejo }), poi(110, { age_s: 1 }) ]};
pintar();
limpiar();
marcar(estado.pois[2], 'no');
pintar();
ok(xs() === '', 'limpiado, caducado y descartado: nada a la vista');
alternarHistorial();
console.log('  historial -> visibles x =', xs(), '| alfa', visibles.map(p => alfaDe(p).toFixed(2)).join(' '),
            '| cabecera:', document.getElementById('cuenta').textContent);
ok(xs() === '90,100', 'historial tiene que mostrar lo que ocultaron las capas, no lo descartado');
ok(visibles.every(p => alfaDe(p) > 0 && alfaDe(p) < 1), 'en historial lo oculto se ve atenuado');
ok(tarjetas().length === 2, 'la lista tambien sigue al historial');
alternarHistorial();
ok(xs() === '', 'apagar historial vuelve a ocultarlos');

// 5. A POI with no age (older firmware) is never faded: there is nothing to fade it by.
estado = { pois: [ poi(200, {}) ]};
delete estado.pois[0].age_s;
pintar();
ok(visibles.length === 1 && alfaDe(estado.pois[0]) === 1, 'sin age_s no hay nada que desvanecer');

// 6. "limpiar" respeta lo confirmado a proposito, asi que sobre un mapa de puros confirmados no
// hace nada visible. Un boton que no hace nada visible se lee como roto: tiene que DECIRLO. El caso
// no es raro, es el del banco con camara en vivo, donde todo lo que se sostiene queda confirmado.
limpiados.length = 0;
estado = { ahora: 1000, pois: [ poi(0, { mature: true, age_s: 1 }), poi(90, { mature: true, age_s: 2 }) ]};
limpiar();
console.log('  mapa de puros confirmados :', document.getElementById('cuenta').textContent);
ok(document.getElementById('cuenta').textContent.indexOf('nada que limpiar') >= 0,
   'limpiar sobre puros confirmados tiene que decir que no hizo nada, y por que');
ok(document.getElementById('cuenta').textContent.indexOf('2 puntos estan confirmados') >= 0,
   'tiene que decir cuantos respeto');
ok(visibles.length === 2, 'limpiar no puede ocultar un confirmado');

limpiados.length = 0;
estado = { ahora: 1000, pois: [ poi(0, { mature: true, age_s: 1 }), poi(90, { mature: false, age_s: 2 }) ]};
limpiar();
console.log('  con uno sin confirmar     :', document.getElementById('cuenta').textContent);
ok(document.getElementById('cuenta').textContent.indexOf('limpiados 1') >= 0,
   'cuando limpia algo tiene que decir cuantos');
`;

console.log('======================================================================');
console.log('CAPAS DEL MAPA: CONTACTOS VIVOS Y PERSISTENTES (codigo real de gs_mapa.py)');
console.log('======================================================================');
eval(codigo + prueba);
console.log();
console.log('TODO OK');

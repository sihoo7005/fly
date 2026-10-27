const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { loadNetwork, simulate } = require('./web-assets/simulator.js');

const bytes = fs.readFileSync('web-assets/network.bin');
const network = loadNetwork(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const model = JSON.parse(fs.readFileSync('web-assets/model.json', 'utf8'));
const features = fs.readFileSync('test_data/game-supervised-features.csv', 'utf8').trim().split(/\r?\n/);
const predictions = fs.readFileSync('test_data/game-supervised-predictions.csv', 'utf8').trim().split(/\r?\n/);
assert.deepEqual(features[0].split(',').slice(1), model.outputIds);
let sample;
for (let i = 1; i < features.length; i++) {
  const [input, ...counts] = features[i].split(',');
  const expected = predictions[i].split(',');
  assert.equal(input, expected[0]);
  const result = simulate(network, model, [...input].map(Number));
  assert.deepEqual(result.outputCounts, counts.map(Number), `output spikes for ${input}`);
  expected[2].split('|').forEach((score, j) => assert.ok(Math.abs(result.scores[j] - Number(score)) < 0.00001, `score for ${input}`));
  assert.equal(['left', 'center', 'right'][result.action], expected[4], `action for ${input}`);
  assert.equal(result.active.reduce((sum, value) => sum + value, 0), result.activeNeurons);
  if (input === '001001') sample = result;
}
assert.equal(features.length, 37);

const html = fs.readFileSync('game.html', 'utf8');
assert.ok(html.indexOf('<section class="card game-card"') < html.indexOf('<section class="card viewer-card"'));
assert.ok(html.indexOf('<section class="card viewer-card"') < html.indexOf('<aside class="side">'));
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
const buttons = [0, 1, 2].map(lane => ({ dataset: { lane: String(lane) }, addEventListener() {}, disabled: false }));
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, { addEventListener() {}, classList: { toggle() {} }, style: {}, width: 528, height: 264,
      getContext() { return { createImageData(w, h) { return { data: new Uint8ClampedArray(w * h * 4) }; }, putImageData(image) { elements.get('neuronMap').image = image.data; } }; } });
    return elements.get(id);
  },
  querySelectorAll() { return buttons; },
  addEventListener() {},
};
class FakeWorker { constructor() { globalThis.fakeWorker = this; this.messages = []; } postMessage(msg) { this.messages.push(msg); } }
const context = { document, Math, Object, Uint8ClampedArray, Worker: FakeWorker, setTimeout, clearTimeout };
vm.runInNewContext(`${script}\nglobalThis.game = { state, inputVector, act, makeRow, flyTurn, setMode, startGame, show3D };`, context);
const { state, act, makeRow, flyTurn, setMode, startGame, show3D } = context.game;
assert.equal((elements.get('actionScores').innerHTML.match(/class="score-line/g) || []).length, 3);
assert.match(elements.get('actionScores').innerHTML, /—/);
for (let i = 0; i < 1000; i++) { const count = makeRow().filter(Boolean).length; assert.ok(count >= 1 && count <= 2); }
const visualNeurons = [{ node: model.inputGroups[2][0], id: model.inputIds[2][0] }, { node: model.outputIndices[0], id: model.outputIds[0] }];
fakeWorker.onmessage({ data: { type: 'ready', neurons: network.neurons, visualNeurons } });
state.rows = [[false, false, true], [false, false, true]];
state.done = false; state.auto = false; state.turn = 0; state.reward = 0;
state.analysis = null;
flyTurn();
assert.equal(state.turn, 0, 'wait for the current calculation');
const stale = state.request;
startGame();
fakeWorker.onmessage({ data: { type: 'result', id: stale, ...sample } });
assert.equal(state.analysis, null, 'discard results from a previous game');
state.rows = [[false, false, true], [false, false, true]];
fakeWorker.onmessage({ data: { type: 'result', id: state.request, ...sample } });
show3D();
assert.equal(elements.get('open3d').href, elements.get('brainView').src);
assert.equal(elements.get('viewPlaceholder').style.display, 'none');
const scene = JSON.parse(decodeURIComponent(elements.get('brainView').src.split('#!')[1]));
assert.equal(scene.layers[1].source, 'precomputed://gs://flywire_v141_m783');
for (const neuron of visualNeurons) {
  assert.equal(scene.layers[1].segmentColors[neuron.id], sample.active[neuron.node] ? '#c6f36b' : '#6b7472');
}
assert.equal(state.snapshotKey, '001001');
const previousMap = elements.get('neuronMap').image;
flyTurn();
assert.equal(state.lane, 0);
assert.equal(state.reward, 1);
assert.equal(state.turn, 1);
assert.equal(state.analysis, null, 'calculate again for the next turn');
assert.match(elements.get('actionScores').innerHTML, /—/, 'hide percentages while calculating');
assert.doesNotMatch(elements.get('actionScores').innerHTML, />[0-9]+\.[0-9]+%</, 'no stale percentages');
assert.deepEqual(elements.get('neuronMap').image, previousMap, 'keep previous neuron map while calculating');
assert.match(elements.get('modelStatus').textContent, /이전 발화 지도/);
assert.match(elements.get('neuronInfo').textContent, /이전 결과/);
flyTurn();
assert.equal(state.turn, 1, 'never act on the previous calculation');
const active = new Uint8Array(network.neurons); active[1] = 1;
fakeWorker.onmessage({ data: { type: 'result', id: state.request, ...sample, probabilities: [0.1, 0.2, 0.7], action: 2, active, activeNeurons: 1, totalSpikes: 1 } });
assert.match(elements.get('actionScores').innerHTML, /70.0%/);
assert.equal(elements.get('neuronMap').image[4], 198, 'show the new neuron map after calculation');
assert.doesNotMatch(elements.get('neuronInfo').textContent, /이전 결과/);
setMode('manual');
state.rows = [[true, true, false], [false, false, true]];
state.done = false; state.turn = 0; state.reward = 0; state.analysis = sample;
act(0);
assert.equal(state.done, true);
assert.equal(state.reward, -3);
assert.match(state.message, /충돌/);
console.log('All 36 network responses match Python; game checks passed.');

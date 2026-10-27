const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { loadNetwork, simulate } = require('./web-assets/simulator.js');

const bytes = fs.readFileSync('web-assets/network.bin');
const network = loadNetwork(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const model = JSON.parse(fs.readFileSync('web-assets/model.json', 'utf8'));
const tinyNetwork = { neurons: 3, offsets: new Uint32Array([0, 2, 2, 2]), targets: new Uint32Array([1, 2]), synapses: new Uint32Array([60, 100]) };
const tinyModel = { inputGroups: [[0]], outputIndices: [1, 2], weights: [[0, 0, 0], [0, 0, 0], [0, 0, 0]], weight: 0.025, scale: 1, steps: 2 };
const tinyResult = simulate(tinyNetwork, tinyModel, [1]);
assert.deepEqual(Array.from(tinyResult.peakExcess), [0, 0.5, 1.5]);
assert.deepEqual(Array.from(tinyResult.active), [1, 1, 1]);
const features = fs.readFileSync('test_data/game-supervised-features.csv', 'utf8').trim().split(/\r?\n/);
const predictions = fs.readFileSync('test_data/game-supervised-predictions.csv', 'utf8').trim().split(/\r?\n/);
assert.deepEqual(features[0].split(',').slice(1), model.outputIds);
assert.equal(model.visualNeurons.length, 50);
assert.equal(new Set(model.visualNeurons.map(n => n.id)).size, 50);
const outputNodes = new Set(model.outputIndices), inputNodes = new Set(model.inputGroups.flat());
assert.equal(model.visualNeurons.filter(n => outputNodes.has(n.node)).length, 35);
assert.equal(model.visualNeurons.filter(n => !outputNodes.has(n.node) && !inputNodes.has(n.node)).length, 15);
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
  assert.equal(result.spikeCounts.reduce((sum, value) => sum + value, 0), result.totalSpikes);
  model.outputIndices.forEach((node, j) => assert.equal(result.spikeCounts[node], result.outputCounts[j]));
  if (input === '001001') sample = result;
}
assert.equal(features.length, 37);

const html = fs.readFileSync('game.html', 'utf8');
assert.ok(html.indexOf('<section class="card game-card"') < html.indexOf('<section class="card viewer-card"'));
assert.ok(html.indexOf('<section class="card viewer-card"') < html.indexOf('<aside class="side">'));
assert.doesNotMatch(html, /보상/);
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
vm.runInNewContext(`${script}\nglobalThis.game = { state, inputVector, act, makeRow, flyTurn, setMode, startGame, show3D, activityColor };`, context);
const { state, act, makeRow, flyTurn, setMode, startGame, show3D, activityColor } = context.game;
assert.equal(activityColor(false, 0), 0x6b7472);
assert.equal(activityColor(true, 0), 0xc6f36b);
assert.equal(activityColor(true, 7.5), 0xe3af64);
assert.equal(activityColor(true, 15), 0xff6b5c);
assert.equal(activityColor(true, 30), 0xff6b5c);
assert.equal((elements.get('actionScores').innerHTML.match(/class="score-line/g) || []).length, 3);
assert.match(elements.get('actionScores').innerHTML, /—/);
for (let i = 0; i < 1000; i++) { const count = makeRow().filter(Boolean).length; assert.ok(count >= 1 && count <= 2); }
const visualNeurons = model.visualNeurons;
fakeWorker.onmessage({ data: { type: 'ready', neurons: network.neurons, visualNeurons } });
state.rows = [[false, false, true], [false, false, true]];
state.done = false; state.auto = false; state.turn = 0;
state.analysis = null;
flyTurn();
assert.equal(state.turn, 0, 'wait for the current calculation');
const stale = state.request;
startGame();
fakeWorker.onmessage({ data: { type: 'result', id: stale, ...sample } });
assert.equal(state.analysis, null, 'discard results from a previous game');
state.rows = [[false, false, true], [false, false, true]];
fakeWorker.onmessage({ data: { type: 'result', id: state.request, ...sample } });
assert.equal(state.snapshotKey, '001001', 'load 3D automatically after calculation');
assert.equal(elements.get('open3d').href, elements.get('brainView').src);
assert.equal(elements.get('viewPlaceholder').style.display, 'none');
const scene = JSON.parse(decodeURIComponent(elements.get('brainView').src.split('#!')[1]));
assert.equal(scene.layers[1].source, 'precomputed://gs://flywire_v141_m783');
for (const neuron of visualNeurons) {
  assert.equal(scene.layers[1].segmentColors[neuron.id], '#'+activityColor(sample.active[neuron.node], sample.peakExcess[neuron.node]).toString(16).padStart(6,'0'));
}
assert.equal(state.snapshotKey, '001001');
const previousMap = elements.get('neuronMap').image;
const previousView = elements.get('brainView').src;
flyTurn();
assert.equal(state.lane, 0);
assert.equal(state.turn, 1);
assert.equal(state.analysis, null, 'calculate again for the next turn');
assert.match(elements.get('actionScores').innerHTML, /—/, 'hide percentages while calculating');
assert.doesNotMatch(elements.get('actionScores').innerHTML, />[0-9]+\.[0-9]+%</, 'no stale percentages');
assert.deepEqual(elements.get('neuronMap').image, previousMap, 'keep previous neuron map while calculating');
assert.equal(elements.get('brainView').src, previousView, 'keep 3D while calculating');
assert.match(elements.get('modelStatus').textContent, /이전 발화 지도/);
assert.match(elements.get('neuronInfo').textContent, /이전 결과/);
flyTurn();
assert.equal(state.turn, 1, 'never act on the previous calculation');
const active = new Uint8Array(network.neurons); active[1] = 1; active[visualNeurons[0].node] = 1;
const peakExcess = new Float64Array(network.neurons); peakExcess[visualNeurons[0].node] = 20;
fakeWorker.onmessage({ data: { type: 'result', id: state.request, ...sample, probabilities: [0.1, 0.2, 0.7], action: 2, active, peakExcess, activeNeurons: 2, totalSpikes: 3 } });
assert.match(elements.get('actionScores').innerHTML, /70.0%/);
assert.notEqual(elements.get('brainView').src, previousView, 'refresh 3D after calculation');
assert.equal(state.snapshotKey, state.rows.flatMap(row => row.map(Number)).join(''));
const nextScene = JSON.parse(decodeURIComponent(elements.get('brainView').src.split('#!')[1]));
for (const neuron of visualNeurons) assert.equal(nextScene.layers[1].segmentColors[neuron.id], '#'+activityColor(active[neuron.node], peakExcess[neuron.node]).toString(16).padStart(6,'0'));
assert.equal(nextScene.layers[1].segmentColors[visualNeurons[0].id], '#ff6b5c', 'large threshold excess is red');
assert.equal(elements.get('neuronMap').image[4], 198, 'show the new neuron map after calculation');
assert.doesNotMatch(elements.get('neuronInfo').textContent, /이전 결과/);
setMode('manual');
state.rows = [[true, true, false], [false, false, true]];
state.done = false; state.turn = 0; state.analysis = sample;
act(0);
assert.equal(state.done, true);
assert.match(state.message, /충돌/);
console.log('All 36 network responses match Python; game checks passed.');

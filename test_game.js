const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync('game.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const policy = html.match(/<script id="policy-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const elements = new Map();
const buttons = [0, 1, 2].map(lane => ({ dataset: { lane: String(lane) }, addEventListener() {}, disabled: false }));
const document = {
  getElementById(id) {
    if (id === 'policy-data') return { textContent: policy };
    if (!elements.has(id)) elements.set(id, { addEventListener() {}, classList: { toggle() {} } });
    return elements.get(id);
  },
  querySelectorAll() { return buttons; },
  addEventListener() {},
};
const context = { document, Math, Object, URL: {}, Blob: function () {}, setTimeout, clearTimeout };
vm.runInNewContext(`${script}\nglobalThis.game = { state, inputVector, act, makeRow, chooseAction, flyTurn, setMode, policy };`, context);
const { state, inputVector, act, makeRow, chooseAction, flyTurn, setMode, policy: decisions } = context.game;

assert.equal(Object.keys(decisions).length, 36);
assert.equal(inputVector().length, 6);
for (let i = 0; i < 1000; i++) { const count = makeRow().filter(Boolean).length; assert.ok(count >= 1 && count <= 2); }
assert.equal(chooseAction([0, 0, 1, 0, 0, 1]), 0);
state.mode = 'fly'; state.rows = [[false, false, true], [false, false, true]];
state.done = false; state.auto = false; state.turn = 0; state.reward = 0;
flyTurn();
assert.equal(state.lane, 0);
assert.equal(state.reward, 1);
assert.equal(state.turn, 1);
setMode('manual');
state.rows = [[true, true, false], [false, false, true]];
state.done = false; state.turn = 0; state.reward = 0;
act(0);
assert.equal(state.done, true);
assert.equal(state.reward, -3);
assert.match(state.message, /충돌/);
console.log('Game and embedded policy checks passed.');

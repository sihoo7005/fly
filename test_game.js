const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync('game.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
const buttons = [0, 1, 2].map(lane => ({ dataset: { lane: String(lane) }, addEventListener() {}, disabled: false }));
const document = {
  getElementById(id) { if (!elements.has(id)) elements.set(id, { addEventListener() {} }); return elements.get(id); },
  querySelectorAll() { return buttons; },
  addEventListener() {},
};
const context = { document, Math, Object, URL: {}, Blob: function () {} };
vm.runInNewContext(`${script}\nglobalThis.game = { state, inputVector, act, makeRow };`, context);
const { state, inputVector, act, makeRow } = context.game;

assert.equal(inputVector().length, 6);
for (let i = 0; i < 1000; i++) { const count = makeRow().filter(Boolean).length; assert.ok(count >= 1 && count <= 2); }
state.rows = [[true, false, true], [false, true, false]];
state.done = false; state.turn = 0; state.score = 0; state.history = [];
act(1);
assert.equal(state.history[0].input, '101010');
assert.equal(state.history[0].reward, 1);
assert.equal(state.lane, 1);
state.rows = [[true, true, false], [false, false, true]];
state.done = false; state.turn = 0; state.score = 0; state.history = [];
act(0);
assert.equal(state.done, true);
assert.equal(state.history[0].reward, -3);
assert.equal(state.history[0].outcome, '충돌');
console.log('Game checks passed.');

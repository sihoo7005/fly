/* Discrete-time connectome simulator matching model.py and the trained readout. */
function loadNetwork(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength < 16 || String.fromCharCode(...new Uint8Array(buffer, 0, 8)) !== 'FLYWEB1\n') throw new Error('Invalid network file');
  const neurons = view.getUint32(8, true), edges = view.getUint32(12, true);
  if (buffer.byteLength !== 16 + (neurons + 1 + edges * 2) * 4) throw new Error('Invalid network size');
  if (new Uint8Array(new Uint32Array([1]).buffer)[0] !== 1) {
    for (let i = 16; i < buffer.byteLength; i += 4) {
      const value = view.getUint32(i, true);
      view.setUint32(i, value, false);
    }
  }
  const offsets = new Uint32Array(buffer, 16, neurons + 1);
  const targets = new Uint32Array(buffer, 16 + (neurons + 1) * 4, edges);
  const synapses = new Uint32Array(buffer, 16 + (neurons + 1 + edges) * 4, edges);
  if (offsets[0] !== 0 || offsets[neurons] !== edges) throw new Error('Invalid network offsets');
  return { neurons, offsets, targets, synapses };
}

function simulate(network, model, bits) {
  const { neurons, offsets, targets, synapses } = network;
  const voltage = new Float64Array(neurons), updated = new Uint8Array(neurons), blocked = new Uint8Array(neurons);
  const outputAt = new Int32Array(neurons); outputAt.fill(-1);
  model.outputIndices.forEach((node, i) => { outputAt[node] = i; });
  const outputCounts = new Uint8Array(model.outputIndices.length), active = new Uint8Array(neurons), spikeCounts = new Uint8Array(neurons);
  let incoming = new Map(), totalSpikes = 0, activeNeurons = 0;
  bits.forEach((bit, i) => { if (bit) model.inputGroups[i].forEach(node => incoming.set(node, 1)); });
  const stepCounts = [];
  for (let step = 0; step < model.steps; step++) {
    const fired = [], next = new Map();
    for (const [node, amount] of incoming) {
      const value = voltage[node] * Math.exp(-(step - updated[node]) / 20) + amount;
      updated[node] = step;
      if (step < blocked[node]) { voltage[node] = 0; continue; }
      if (value >= 1) {
        fired.push(node); voltage[node] = 0; blocked[node] = step + 2;
        spikeCounts[node]++;
        if (!active[node]) { active[node] = 1; activeNeurons++; }
        if (outputAt[node] !== -1) outputCounts[outputAt[node]]++;
      } else voltage[node] = value;
    }
    stepCounts.push(fired.length); totalSpikes += fired.length;
    for (const node of fired) {
      for (let edge = offsets[node]; edge < offsets[node + 1]; edge++) {
        const target = targets[edge];
        next.set(target, (next.get(target) || 0) + model.weight * synapses[edge]);
      }
    }
    incoming = next;
  }
  const scores = model.weights.map(row => {
    let score = row[outputCounts.length];
    for (let i = 0; i < outputCounts.length; i++) if (outputCounts[i]) score += row[i] * outputCounts[i] / model.scale;
    return score;
  });
  const best = Math.max(...scores), exponentials = scores.map(value => Math.exp(value - best));
  const sum = exponentials.reduce((a, b) => a + b, 0);
  return { scores, probabilities: exponentials.map(value => value / sum), action: scores.indexOf(best),
    outputCounts: Array.from(outputCounts), active, spikeCounts, activeNeurons, totalSpikes, stepCounts };
}

if (typeof module !== 'undefined') module.exports = { loadNetwork, simulate };
if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  let network, model;
  Promise.all([fetch('network.bin'), fetch('model.json?v=spikes2')])
    .then(async responses => {
      if (responses.some(response => !response.ok)) throw new Error('모델 파일을 불러오지 못했습니다.');
      network = loadNetwork(await responses[0].arrayBuffer()); model = await responses[1].json();
      const visualNeurons = model.inputGroups.flatMap((group, g) => group.slice(0, 2).map((node, i) => ({ node, id: model.inputIds[g][i] })));
      for (let i = 0; i < 20; i++) {
        const index = Math.floor(i * (model.outputIndices.length - 1) / 19);
        visualNeurons.push({ node: model.outputIndices[index], id: model.outputIds[index] });
      }
      visualNeurons.push(...model.visualNeurons);
      self.postMessage({ type: 'ready', neurons: network.neurons, visualNeurons });
    })
    .catch(error => self.postMessage({ type: 'error', message: error.message }));
  self.onmessage = event => {
    try {
      const { id, bits } = event.data;
      self.postMessage({ type: 'result', id, ...simulate(network, model, bits) });
    } catch (error) { self.postMessage({ type: 'error', message: error.message }); }
  };
}

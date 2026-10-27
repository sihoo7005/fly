"""Rank neurons by visible threshold-excess variation across game inputs."""
import json
import math
import time
from pathlib import Path
from model import Network, Parameters, Simulator
from experiments.game_supervised import game_states

ROOT = Path(__file__).resolve().parents[1]
COLOR_MAX = 50.0

def main():
    network = Network.load(ROOT / 'results/princeton-network.bin')
    manifest = json.loads((ROOT / 'experiments/neuron_groups.json').read_text())
    groups = manifest['input_groups']
    input_ids = {int(root) for group in groups['groups'].values() for root in group['root_ids']}
    descending_ids = {int(root) for root in manifest['output_group']['root_ids']}
    readout = json.loads((ROOT / 'results/game-supervised-weights.json').read_text())
    readout_ids = {int(root) for root in readout['input_root_ids']}
    candidates = {
        'descending': [network.index[root] for root in descending_ids if root in network.index],
        'intermediate': [node for node, root in enumerate(network.ids)
                         if root not in input_ids | descending_ids | readout_ids],
    }
    signal_groups = [groups['groups'][name]['root_ids'] for name in groups['signal_order']]
    states = game_states()
    n = len(network.ids)
    sums, squares = [0.0] * n, [0.0] * n
    mins, maxs, raw_maxs = [COLOR_MAX] * n, [0.0] * n, [0.0] * n
    started = time.monotonic()
    for state_index, bits in enumerate(states, 1):
        sim = Simulator(network, Parameters(weight=0.025))
        stimulus = {root: 1.0 for bit, ids in zip(bits, signal_groups) if bit for root in ids}
        peaks = {}
        for step in range(7):
            incoming, sim.pending = sim.pending, {}
            if step == 0:
                for root, amount in stimulus.items():
                    node = network.index[root]
                    incoming[node] = incoming.get(node, 0.0) + amount
            fired, p = [], sim.parameters
            for node, amount in incoming.items():
                elapsed = (sim.time_step - sim.updated[node]) * p.dt_ms
                value = sim.voltage[node] * math.exp(-elapsed / p.tau_ms) + amount
                sim.updated[node] = sim.time_step
                if sim.time_step < sim.blocked_until[node]:
                    sim.voltage[node] = 0
                    continue
                if value >= p.threshold:
                    peaks[node] = max(peaks.get(node, 0.0), value - p.threshold)
                    fired.append(node)
                    sim.voltage[node] = 0
                    sim.blocked_until[node] = sim.time_step + sim.refractory_steps
                else:
                    sim.voltage[node] = value
            for source in fired:
                coefficient = p.weight * sim.signs[source]
                for edge in range(network.offsets[source], network.offsets[source + 1]):
                    target = network.targets[edge]
                    sim.pending[target] = sim.pending.get(target, 0.0) + coefficient * network.synapses[edge]
            sim.time_step += 1
        for node, raw in peaks.items():
            excess = min(COLOR_MAX, raw)
            sums[node] += excess
            squares[node] += excess * excess
            mins[node] = min(mins[node], excess)
            maxs[node] = max(maxs[node], excess)
            raw_maxs[node] = max(raw_maxs[node], raw)
        if state_index % 12 == 0:
            print(f'{state_index}/{len(states)} inputs in {time.monotonic() - started:.1f}s', flush=True)

    output = {
        'description': ('35 descending and 15 intermediate neurons ranked by standard deviation '
                        'of peak threshold excess clipped to the displayed 0–50 color scale '
                        'across all 36 valid game inputs. Visualization sample only; not representative.'),
        'selection': {'inputs': len(states),
                      'metric': 'population standard deviation of per-input peak threshold excess clipped to 0–50 (display color scale)',
                      'descending_candidates': len(candidates['descending']),
                      'intermediate_candidates': len(candidates['intermediate'])},
        'descending': [], 'intermediate': [],
    }
    for name, nodes in candidates.items():
        count = 35 if name == 'descending' else 15
        ranked = []
        for node in nodes:
            mean = sums[node] / len(states)
            variance = max(0.0, squares[node] / len(states) - mean * mean)
            ranked.append((math.sqrt(variance), maxs[node] - mins[node], raw_maxs[node], node))
        ranked.sort(reverse=True)
        output[name] = [
            {'root_id': str(network.ids[node]), 'color_excess_sd': round(sd, 6),
             'color_excess_span': round(span, 6), 'max_raw_excess': round(raw, 6)}
            for sd, span, raw, node in ranked[:count]
        ]
        print(name, [(item['root_id'], item['color_excess_sd']) for item in output[name]])
    target = ROOT / 'experiments/visual_neurons.json'
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote {target} in {time.monotonic() - started:.1f}s')

if __name__ == '__main__':
    main()

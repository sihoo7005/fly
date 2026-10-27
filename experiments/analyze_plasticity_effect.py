"""Compare baseline and reward-trained synapses on all valid game inputs."""
import bisect
import csv
import itertools
import json
import time
from pathlib import Path

from model import Network, Parameters, Simulator

ROOT = Path(__file__).resolve().parents[1]
LANES = ('left', 'center', 'right')


def main():
    manifest = json.loads((ROOT / 'experiments/neuron_groups.json').read_text())
    network = Network.load(ROOT / 'results/princeton-network.bin')
    signal_names = manifest['input_groups']['signal_order']
    input_groups = [manifest['input_groups']['groups'][name]['root_ids'] for name in signal_names]
    output_ids = sorted(manifest['output_group']['root_ids'])
    output_nodes = [network.index[root] for root in output_ids]
    action_of = {node: i % 3 for i, node in enumerate(output_nodes)}

    learned_pairs = {}
    with (ROOT / 'results/game-plasticity-wide-weights.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            learned_pairs[(int(row['pre_root_id']), int(row['post_root_id']))] = float(row['multiplier'])
    factors = {}
    for source in range(len(network.ids)):
        pre = network.ids[source]
        for edge in range(network.offsets[source], network.offsets[source + 1]):
            key = (pre, network.ids[network.targets[edge]])
            if key in learned_pairs:
                factors[edge] = learned_pairs.pop(key)
    if learned_pairs:
        raise ValueError(f'{len(learned_pairs)} learned connections are absent from network')

    def response(bits, multipliers):
        sim = Simulator(network, Parameters(weight=0.025), edge_multipliers=multipliers)
        stimulus = {root: 1.0 for bit, ids in zip(bits, input_groups) if bit for root in ids}
        scores = [0, 0, 0]
        total = 0
        for step in range(7):
            fired = sim.step(stimulus if step == 0 else {})
            total += len(fired)
            for root in fired:
                action = action_of.get(network.index[root])
                if action is not None:
                    scores[action] += 1
        top = [i for i, score in enumerate(scores) if score == max(scores)]
        safe = [i for i, blocked in enumerate(bits[:3]) if blocked == 0]
        safe_chance = sum(i in safe for i in top) / len(top)
        return scores, total, top, safe_chance

    rows = [row for row in itertools.product((0, 1), repeat=3) if sum(row) in (1, 2)]
    states = [tuple(near + far) for near in rows for far in rows]
    records = []
    start = time.monotonic()
    for bits in states:
        before = response(bits, None)
        after = response(bits, factors)
        records.append({'input': ''.join(map(str, bits)),
                        'safe_lanes': '|'.join(LANES[i] for i in range(3) if not bits[i]),
                        'base_scores': '|'.join(map(str, before[0])),
                        'learned_scores': '|'.join(map(str, after[0])),
                        'base_top_actions': '|'.join(LANES[i] for i in before[2]),
                        'learned_top_actions': '|'.join(LANES[i] for i in after[2]),
                        'base_safe_chance': before[3], 'learned_safe_chance': after[3],
                        'base_network_spikes': before[1], 'learned_network_spikes': after[1],
                        'score_changed': before[0] != after[0],
                        'top_action_changed': before[2] != after[2]})

    out = ROOT / 'results/plasticity-effect.csv'
    with out.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    changed = [r for r in records if r['score_changed']]
    switched = [r for r in records if r['top_action_changed']]
    result = {'states': len(records), 'learned_connections': len(factors),
              'states_with_changed_scores': len(changed), 'states_with_changed_top_actions': len(switched),
              'mean_safe_choice_before': round(sum(r['base_safe_chance'] for r in records) / len(records), 3),
              'mean_safe_choice_after': round(sum(r['learned_safe_chance'] for r in records) / len(records), 3),
              'forced_collision_states_before': sum(r['base_safe_chance'] == 0 for r in records),
              'forced_collision_states_after': sum(r['learned_safe_chance'] == 0 for r in records),
              'mean_network_spikes_before': round(sum(r['base_network_spikes'] for r in records) / len(records), 1),
              'mean_network_spikes_after': round(sum(r['learned_network_spikes'] for r in records) / len(records), 1),
              'seconds': round(time.monotonic() - start, 1), 'detail_csv': str(out.relative_to(ROOT))}
    (ROOT / 'results/plasticity-effect.summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

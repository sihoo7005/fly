"""Train FlyWire synaptic multipliers from rewards in the three-lane game."""
import argparse
import bisect
import csv
import json
import math
import random
import time
from pathlib import Path

from model import Network, Parameters, Simulator
from experiments.game_connectome import LANES, make_row

ROOT = Path(__file__).resolve().parents[1]


def summarize(records, phase):
    episodes = {}
    for row in records:
        if row['phase'] == phase:
            episodes.setdefault(row['episode'], []).append(row)
    if not episodes:
        return {'episodes': 0}
    wins = sum(rows[-1]['outcome'] == 'finish' for rows in episodes.values())
    rewards = [sum(row['reward'] for row in rows) for rows in episodes.values()]
    count = len(episodes)
    return {'episodes': count, 'finished': wins, 'finish_rate': round(wins / count, 3),
            'mean_reward': round(sum(rewards) / count, 3),
            'mean_turns': round(sum(map(len, episodes.values())) / count, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-episodes', type=int, default=20)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=7005)
    parser.add_argument('--weight', type=float, default=0.025)
    parser.add_argument('--window-ms', type=int, default=7)
    parser.add_argument('--learning-rate', type=float, default=0.03)
    parser.add_argument('--min-multiplier', type=float, default=0.5)
    parser.add_argument('--max-multiplier', type=float, default=1.5)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--out', default='results/game-plasticity.csv')
    args = parser.parse_args()
    if args.train_episodes < 0 or args.eval_episodes < 1 or args.window_ms < 1:
        parser.error('training episodes must be nonnegative; evaluation episodes and window must be positive')
    if args.weight <= 0 or args.learning_rate <= 0 or not 0 <= args.epsilon <= 1:
        parser.error('weight and learning rate must be positive; epsilon must be in [0, 1]')
    if not 0 < args.min_multiplier < 1 < args.max_multiplier:
        parser.error('multiplier bounds must straddle 1')

    manifest = json.loads((ROOT / 'experiments/neuron_groups.json').read_text())
    network = Network.load(ROOT / 'results/princeton-network.bin')
    signal_names = manifest['input_groups']['signal_order']
    input_groups = [manifest['input_groups']['groups'][name]['root_ids'] for name in signal_names]
    output_ids = sorted(manifest['output_group']['root_ids'])
    output_nodes = [network.index[root] for root in output_ids]
    node_action = {node: i % 3 for i, node in enumerate(output_nodes)}
    output_set = set(output_nodes)
    incoming_output = {node: [] for node in output_nodes}
    for source in range(len(network.ids)):
        for edge in range(network.offsets[source], network.offsets[source + 1]):
            target = network.targets[edge]
            if target in output_set:
                incoming_output[target].append((edge, source))
    factors = {}
    records = []
    sim_seconds = 0.0
    updated_total = 0

    def observe(bits):
        nonlocal sim_seconds
        sim = Simulator(network, Parameters(weight=args.weight), edge_multipliers=factors)
        stimulus = {root: 1.0 for bit, ids in zip(bits, input_groups) if bit for root in ids}
        scores = [0, 0, 0]
        history = []
        network_spikes = 0
        max_step_spikes = 0
        started = time.monotonic()
        for step in range(args.window_ms):
            fired = {network.index[root] for root in sim.step(stimulus if step == 0 else {})}
            history.append(fired)
            network_spikes += len(fired)
            max_step_spikes = max(max_step_spikes, len(fired))
            for node in fired:
                action = node_action.get(node)
                if action is not None:
                    scores[action] += 1
        eligible = [set(), set(), set()]
        for previous, current in zip(history, history[1:]):
            for target in current & output_set:
                for edge, source in incoming_output[target]:
                    if source in previous:
                        eligible[node_action[target]].add(edge)
        sim_seconds += time.monotonic() - started
        return scores, eligible, network_spikes, max_step_spikes

    rng = random.Random(args.seed)

    def episode(seed, phase, plastic):
        nonlocal updated_total
        game_rng = random.Random(seed)
        rows = [make_row(game_rng), make_row(game_rng)]
        for turn in range(1, 21):
            bits = tuple(rows[0] + rows[1])
            scores, eligible, network_spikes, max_step_spikes = observe(bits)
            if plastic and rng.random() < args.epsilon:
                action = rng.randrange(3)
            else:
                top = max(scores)
                action = rng.choice([i for i, value in enumerate(scores) if value == top])
            crash = bool(rows[0][action])
            reward = -3 if crash else 1
            outcome = 'crash' if crash else 'survive'
            if not crash:
                rows = [rows[1], make_row(game_rng)]
                if turn == 20:
                    reward, outcome = 11, 'finish'
            changed = 0
            if plastic:
                direction = 1 if reward > 0 else -1
                for edge in eligible[action]:
                    value = min(args.max_multiplier, max(args.min_multiplier, factors.get(edge, 1.0) * math.exp(args.learning_rate * direction)))
                    factors[edge] = value
                    changed += 1
                updated_total += changed
            records.append({'phase': phase, 'episode': seed, 'turn': turn,
                            'input': ''.join(map(str, bits)), 'lane': LANES[action],
                            'outcome': outcome, 'reward': reward,
                            'action_spikes_left': scores[0], 'action_spikes_center': scores[1],
                            'action_spikes_right': scores[2], 'network_spikes': network_spikes,
                            'max_step_spikes': max_step_spikes, 'eligible_edges': sum(map(len, eligible)),
                            'updated_edges': changed})
            if crash or turn == 20:
                break

    for episode_index in range(args.train_episodes):
        episode(args.seed + episode_index, 'train', True)
        if (episode_index + 1) % 10 == 0:
            print(f"trained {episode_index + 1}/{args.train_episodes} episodes", flush=True)
    for episode_index in range(args.eval_episodes):
        seed = args.seed + 100_000 + episode_index
        episode(seed, 'plasticity', False)
        # Same game seed, random lane choice; weights stay frozen.
        game_rng = random.Random(seed)
        rows = [make_row(game_rng), make_row(game_rng)]
        rrng = random.Random(seed + 1_000_000)
        for turn in range(1, 21):
            bits = tuple(rows[0] + rows[1])
            scores, eligible, network_spikes, max_step_spikes = observe(bits)
            action = rrng.randrange(3)
            crash = bool(rows[0][action])
            reward = -3 if crash else 1
            outcome = 'crash' if crash else 'survive'
            if not crash:
                rows = [rows[1], make_row(game_rng)]
                if turn == 20:
                    reward, outcome = 11, 'finish'
            records.append({'phase': 'random', 'episode': seed, 'turn': turn,
                            'input': ''.join(map(str, bits)), 'lane': LANES[action],
                            'outcome': outcome, 'reward': reward,
                            'action_spikes_left': scores[0], 'action_spikes_center': scores[1],
                            'action_spikes_right': scores[2], 'network_spikes': network_spikes,
                            'max_step_spikes': max_step_spikes, 'eligible_edges': sum(map(len, eligible)),
                            'updated_edges': 0})
            if crash or turn == 20:
                break
        if (episode_index + 1) % 5 == 0:
            print(f"evaluated {episode_index + 1}/{args.eval_episodes} episodes", flush=True)

    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ('phase', 'episode', 'turn', 'input', 'lane', 'outcome', 'reward',
              'action_spikes_left', 'action_spikes_center', 'action_spikes_right',
              'network_spikes', 'max_step_spikes', 'eligible_edges', 'updated_edges')
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    weight_file = output.with_name(output.stem + '-weights.csv')
    with weight_file.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('pre_root_id', 'post_root_id', 'base_synapses', 'multiplier', 'action_channel'))
        for edge, factor in sorted(factors.items()):
            source = network.ids[bisect.bisect_right(network.offsets, edge) - 1]
            target = network.ids[network.targets[edge]]
            writer.writerow((source, target, network.synapses[edge], f'{factor:.8g}',
                             LANES[node_action[network.targets[edge]]]))

    result = {'settings': {'weight': args.weight, 'window_ms': args.window_ms,
                           'learning_rate': args.learning_rate, 'epsilon': args.epsilon,
                           'min_multiplier': args.min_multiplier, 'max_multiplier': args.max_multiplier,
                           'updated_connectome_edges': len(factors),
                           'reset_network_each_observation': True,
                           'action_channels': 'descending_CV neurons partitioned by sorted root ID modulo 3'},
              'training': summarize(records, 'train'),
              'plasticity_evaluation': summarize(records, 'plasticity'),
              'random_baseline': summarize(records, 'random'),
              'effective_edge_updates': updated_total,
              'simulation_seconds': round(sim_seconds, 2),
              'event_log': str(output.relative_to(ROOT)),
              'learned_weights': str(weight_file.relative_to(ROOT))}
    output.with_suffix('.summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""Run the three-lane game with FlyWire inputs and a reward-trained readout."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

from model import Network, Parameters, Simulator

ROOT = Path(__file__).resolve().parents[1]
LANES = ('left', 'center', 'right')


def make_row(rng):
    blocked = set(rng.sample(range(3), rng.randint(1, 2)))
    return [int(lane in blocked) for lane in range(3)]


def play(seed, choose, features, update=None):
    rng = random.Random(seed)
    rows = [make_row(rng), make_row(rng)]
    events = []
    for turn in range(1, 21):
        bits = tuple(rows[0] + rows[1])
        activity = features(bits)
        lane = choose(activity)
        crash = bool(rows[0][lane])
        reward, outcome = (-3, 'crash') if crash else (1, 'survive')
        if not crash:
            rows = [rows[1], make_row(rng)]
            if turn == 20:
                reward, outcome = 11, 'finish'
        events.append({'seed': seed, 'turn': turn, 'input': ''.join(map(str, bits)),
                       'lane': LANES[lane], 'outcome': outcome, 'reward': reward,
                       'descending_spikes': sum(activity),
                       'active_descending_neurons': sum(x > 0 for x in activity)})
        if update:
            update(activity, lane, reward)
        if crash or turn == 20:
            break
    return events


class Readout:
    def __init__(self, rng, rate, epsilon, n):
        self.weights = [[0.0] * n for _ in LANES]
        self.rng, self.rate, self.epsilon = rng, rate, epsilon
        self.baseline = 0.0
        self.decisions = 0

    def choose(self, activity, training=False):
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(3)
        scores = [sum(w * x for w, x in zip(row, activity)) for row in self.weights]
        best = max(scores)
        return self.rng.choice([i for i, score in enumerate(scores) if abs(score-best) < 1e-12])

    def update(self, activity, action, reward):
        total = sum(activity)
        if total:
            factor = self.rate * (reward - self.baseline) / total
            for i, count in enumerate(activity):
                if count:
                    self.weights[action][i] += factor * count
        self.decisions += 1
        self.baseline += (reward - self.baseline) / min(self.decisions, 20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-episodes', type=int, default=20)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=7005)
    parser.add_argument('--weight', type=float, default=0.02)
    parser.add_argument('--window-ms', type=int, default=7)
    parser.add_argument('--learning-rate', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--out', default='results/game-connectome.csv')
    args = parser.parse_args()
    if args.train_episodes < 0 or args.eval_episodes < 1 or args.window_ms < 1:
        parser.error('training episodes must be nonnegative; evaluation episodes and window must be positive')

    manifest = json.loads((ROOT / 'experiments/neuron_groups.json').read_text())
    network = Network.load(ROOT / 'results/princeton-network.bin')
    names = manifest['input_groups']['signal_order']
    groups = [manifest['input_groups']['groups'][name]['root_ids'] for name in names]
    outputs = sorted(manifest['output_group']['root_ids'])
    node_to_output = {network.index[root]: i for i, root in enumerate(outputs)}
    cache, simulated_seconds = {}, 0.0

    def features(bits):
        nonlocal simulated_seconds
        if bits not in cache:
            sim = Simulator(network, Parameters(weight=args.weight))
            stimulus = {root: 1.0 for bit, ids in zip(bits, groups) if bit for root in ids}
            counts = [0] * len(outputs)
            start = time.monotonic()
            for step in range(args.window_ms):
                for root in sim.step(stimulus if step == 0 else {}):
                    i = node_to_output.get(network.index[root])
                    if i is not None:
                        counts[i] += 1
            simulated_seconds += time.monotonic() - start
            cache[bits] = counts
        return cache[bits]

    rng = random.Random(args.seed)
    policy = Readout(rng, args.learning_rate, args.epsilon, len(outputs))
    records = []
    for episode in range(args.train_episodes):
        events = play(args.seed + episode, lambda x: policy.choose(x, True), features, policy.update)
        records.extend({'phase': 'train', 'episode': episode + 1, **e} for e in events)
    for episode in range(args.eval_episodes):
        seed = args.seed + 100_000 + episode
        events = play(seed, policy.choose, features)
        records.extend({'phase': 'readout', 'episode': episode + 1, **e} for e in events)
        rrng = random.Random(seed + 1_000_000)
        events = play(seed, lambda _: rrng.randrange(3), features)
        records.extend({'phase': 'random', 'episode': episode + 1, **e} for e in events)

    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ('phase', 'episode', 'seed', 'turn', 'input', 'lane', 'outcome', 'reward',
              'descending_spikes', 'active_descending_neurons')
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    def summary(phase):
        episodes = {}
        for event in records:
            if event['phase'] == phase:
                episodes.setdefault(event['episode'], []).append(event)
        if not episodes:
            return {'episodes': 0}
        wins = sum(rows[-1]['outcome'] == 'finish' for rows in episodes.values())
        rewards = [sum(e['reward'] for e in rows) for rows in episodes.values()]
        count = len(episodes)
        return {'episodes': count, 'finished': wins, 'finish_rate': round(wins/count, 3),
                'mean_reward': round(sum(rewards)/count, 3),
                'mean_turns': round(sum(map(len, episodes.values()))/count, 2)}

    result = {'settings': {'weight': args.weight, 'window_ms': args.window_ms,
                           'reset_network_each_observation': True, 'cached_input_patterns': len(cache)},
              'training': summary('train'), 'readout_evaluation': summary('readout'),
              'random_baseline': summary('random'), 'simulation_seconds': round(simulated_seconds, 2),
              'event_log': str(output.relative_to(ROOT))}
    summary_path = output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

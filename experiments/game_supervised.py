"""Train a three-action readout from all valid game states."""
import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path

from model import Network, Parameters, Simulator

ROOT = Path(__file__).resolve().parents[1]
LANES = ('left', 'center', 'right')


def game_states():
    rows = [row for row in itertools.product((0, 1), repeat=3) if sum(row) in (1, 2)]
    return [tuple(near + far) for near in rows for far in rows]


def safe_actions(bits):
    return [i for i, blocked in enumerate(bits[:3]) if not blocked]


def softmax(scores):
    values = [math.exp(score - max(scores)) for score in scores]
    return [value / sum(values) for value in values]


def train(features, targets, width, epochs=2000, learning_rate=1.0, l2=0.001):
    weights = [[0.0] * (width + 1) for _ in LANES]
    for _ in range(epochs):
        gradient = [[0.0] * (width + 1) for _ in LANES]
        for active, target in zip(features, targets):
            scores = [row[-1] + sum(row[i] * x for i, x in active.items()) for row in weights]
            probs = softmax(scores)
            for action in range(3):
                error = probs[action] - target[action]
                gradient[action][-1] += error
                for i, x in active.items():
                    gradient[action][i] += error * x
        scale = learning_rate / len(features)
        for action in range(3):
            for i in range(width):
                weights[action][i] -= scale * (gradient[action][i] + l2 * weights[action][i])
            weights[action][-1] -= scale * gradient[action][-1]
    return weights


def predict(weights, active):
    scores = [row[-1] + sum(row[i] * x for i, x in active.items()) for row in weights]
    return scores, softmax(scores)


def get_features(network, input_groups, output_ids, args):
    path = ROOT / args.features
    meta_path = path.with_suffix('.meta.json')
    states = game_states()
    expected = {'weight': args.weight, 'window_ms': args.window_ms, 'input_groups': input_groups, 'output_ids': output_ids}
    if path.exists() and meta_path.exists() and not args.refresh_features:
        with meta_path.open() as stream:
            cache_matches = json.load(stream) == expected
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if cache_matches and reader.fieldnames == ['input', *map(str, output_ids)]:
                rows = {tuple(map(int, row['input'])): [int(row[str(root)]) for root in output_ids]
                        for row in reader}
                if set(rows) == set(states):
                    return states, [rows[state] for state in states], 0.0
    started = time.monotonic()
    records = []
    for bits in states:
        sim = Simulator(network, Parameters(weight=args.weight))
        stimulus = {root: 1.0 for bit, ids in zip(bits, input_groups) if bit for root in ids}
        counts = {root: 0 for root in output_ids}
        for step in range(args.window_ms):
            for root in sim.step(stimulus if step == 0 else {}):
                if root in counts:
                    counts[root] += 1
        records.append([counts[root] for root in output_ids])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['input', *output_ids])
        writer.writerows([''.join(map(str, bits)), *values] for bits, values in zip(states, records))
    meta_path.write_text(json.dumps(expected) + '\n')
    return states, records, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight', type=float, default=0.025)
    parser.add_argument('--window-ms', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--l2', type=float, default=0.001)
    parser.add_argument('--features', default='results/game-supervised-features.csv')
    parser.add_argument('--weights-out', default='results/game-supervised-weights.json')
    parser.add_argument('--predictions-out', default='results/game-supervised-predictions.csv')
    parser.add_argument('--refresh-features', action='store_true')
    args = parser.parse_args()
    if args.weight <= 0 or args.window_ms < 1 or args.epochs < 1 or args.learning_rate <= 0 or args.l2 < 0:
        parser.error('weight, epochs, and learning rate must be positive; window must be positive; L2 must be nonnegative')

    manifest = json.loads((ROOT / 'experiments/neuron_groups.json').read_text())
    network = Network.load(ROOT / 'results/princeton-network.bin')
    names = manifest['input_groups']['signal_order']
    groups = [manifest['input_groups']['groups'][name]['root_ids'] for name in names]
    output_ids = sorted(manifest['output_group']['root_ids'])
    states, counts, feature_seconds = get_features(network, groups, output_ids, args)
    scale = max(1, max(max(row) for row in counts))
    features = [{i: value / scale for i, value in enumerate(row) if value}
                for row in counts]
    targets = []
    for bits in states:
        safe = safe_actions(bits)
        target = [0.0] * 3
        for action in safe:
            target[action] = 1 / len(safe)
        targets.append(target)
    started = time.monotonic()
    weights = train(features, targets, len(output_ids), args.epochs, args.learning_rate, args.l2)
    training_seconds = time.monotonic() - started

    rows, safe_chances, safe_probs = [], [], []
    for bits, active in zip(states, features):
        scores, probs = predict(weights, active)
        top = [i for i, score in enumerate(scores) if abs(score - max(scores)) < 1e-12]
        safe = safe_actions(bits)
        chance = sum(i in safe for i in top) / len(top)
        mass = sum(probs[i] for i in safe)
        safe_chances.append(chance)
        safe_probs.append(mass)
        rows.append((bits, safe, scores, probs, top))

    weights_path = ROOT / args.weights_out
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(json.dumps({
        'actions': LANES, 'input_root_ids': output_ids,
        'feature_scale': scale, 'bias_index': 'last value in each action row',
        'weights': weights,
        'settings': {'connectome_weight': args.weight, 'window_ms': args.window_ms,
                     'epochs': args.epochs, 'learning_rate': args.learning_rate, 'l2': args.l2},
    }, separators=(',', ':')) + '\n')
    pred_path = ROOT / args.predictions_out
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('input', 'safe_actions', 'scores', 'probabilities', 'top_actions',
                         'top1_safe_chance', 'safe_probability_mass'))
        for bits, safe, scores, probs, top in rows:
            writer.writerow((''.join(map(str, bits)), '|'.join(LANES[i] for i in safe),
                             '|'.join(f'{x:.6g}' for x in scores),
                             '|'.join(f'{x:.6f}' for x in probs),
                             '|'.join(LANES[i] for i in top),
                             sum(i in safe for i in top) / len(top),
                             sum(probs[i] for i in safe)))

    baseline = sum(len(safe_actions(bits)) / 3 for bits in states) / len(states)
    result = {'states': len(states), 'features': len(output_ids),
              'training_scope': 'all 36 valid game inputs; exhaustive domain, not held-out test data',
              'baseline_safe_probability': round(baseline, 4),
              'trained_safe_probability': round(sum(safe_probs) / len(states), 4),
              'top_action_safe_chance': round(sum(safe_chances) / len(states), 4),
              'states_with_safe_top_action': sum(x > 0 for x in safe_chances),
              'feature_generation_seconds': round(feature_seconds, 2),
              'training_seconds': round(training_seconds, 2),
              'weights': str(weights_path.relative_to(ROOT)),
              'predictions': str(pred_path.relative_to(ROOT)),
              'features_cache': str((ROOT / args.features).relative_to(ROOT))}
    weights_path.with_suffix('.summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

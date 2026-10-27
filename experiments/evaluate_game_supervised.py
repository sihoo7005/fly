"""Evaluate a saved supervised readout on new, seeded game episodes."""
import argparse
import csv
import json
import random
from pathlib import Path

from experiments.game_connectome import LANES, make_row
from experiments.game_supervised import game_states, predict, safe_actions

ROOT = Path(__file__).resolve().parents[1]


def load_policy(weights_path, features_path):
    bundle = json.loads(weights_path.read_text())
    output_ids = bundle['input_root_ids']
    scale = bundle['feature_scale']
    expected = {''.join(map(str, bits)) for bits in game_states()}
    with features_path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['input', *map(str, output_ids)]:
            raise ValueError('Feature cache does not match the saved readout')
        features = {row['input']: {i: int(row[str(root)]) / scale
                                   for i, root in enumerate(output_ids)
                                   if int(row[str(root)])}
                    for row in reader}
    if set(features) != expected:
        raise ValueError('Feature cache must contain all 36 valid game inputs')
    return bundle['weights'], features


def play(seed, weights, features, policy):
    game_rng = random.Random(seed)
    tie_rng = random.Random(seed + 2_000_000)
    action_rng = random.Random(seed + 1_000_000)
    rows = [make_row(game_rng), make_row(game_rng)]
    decisions = safe_count = total_reward = 0
    outcome = 'crash'
    for turn in range(1, 21):
        bits = tuple(rows[0] + rows[1])
        if policy == 'supervised':
            scores, _ = predict(weights, features[''.join(map(str, bits))])
            best = max(scores)
            action = tie_rng.choice([i for i, score in enumerate(scores) if abs(score - best) < 1e-12])
        else:
            action = action_rng.randrange(3)
        safe = action in safe_actions(bits)
        decisions += 1
        safe_count += safe
        if not safe:
            total_reward -= 3
            outcome = 'crash'
            break
        total_reward += 1
        rows = [rows[1], make_row(game_rng)]
        if turn == 20:
            total_reward += 10
            outcome = 'finish'
    return {'seed': seed, 'policy': policy, 'outcome': outcome, 'turns': decisions,
            'safe_actions': safe_count, 'decision_accuracy': safe_count / decisions,
            'reward': total_reward}


def summarize(rows):
    episodes = len(rows)
    decisions = sum(row['turns'] for row in rows)
    safe = sum(row['safe_actions'] for row in rows)
    return {'episodes': episodes,
            'decision_accuracy': round(safe / decisions, 4),
            'safe_decisions': safe, 'decisions': decisions,
            'finished': sum(row['outcome'] == 'finish' for row in rows),
            'finish_rate': round(sum(row['outcome'] == 'finish' for row in rows) / episodes, 4),
            'mean_reward': round(sum(row['reward'] for row in rows) / episodes, 4),
            'mean_turns': round(decisions / episodes, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=2_000_000)
    parser.add_argument('--weights', default='results/game-supervised-weights.json')
    parser.add_argument('--features', default='results/game-supervised-features.csv')
    parser.add_argument('--out', default='results/game-supervised-evaluation.csv')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    weights, features = load_policy(ROOT / args.weights, ROOT / args.features)
    results = []
    for i in range(args.episodes):
        seed = args.seed + i
        results.append(play(seed, weights, features, 'supervised'))
        results.append(play(seed, weights, features, 'random'))
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    summary = {'episodes_per_policy': args.episodes, 'seed_start': args.seed,
               'supervised': summarize([row for row in results if row['policy'] == 'supervised']),
               'random_baseline': summarize([row for row in results if row['policy'] == 'random']),
               'evaluation': 'New game seeds; report whether the chosen lane was safe at each decision.',
               'episode_log': str(output.relative_to(ROOT))}
    output.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

import unittest

from experiments.game_supervised import game_states, predict, safe_actions, train


class SupervisedGameTests(unittest.TestCase):
    def test_complete_game_states_have_safe_labels(self):
        states = game_states()
        self.assertEqual(len(states), 36)
        self.assertTrue(all(1 <= len(safe_actions(bits)) <= 2 for bits in states))

    def test_readout_learns_a_simple_three_action_mapping(self):
        features = [{0: 1.0}, {1: 1.0}, {2: 1.0}]
        targets = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        weights = train(features, targets, 3, epochs=300, learning_rate=0.01)
        for active, expected in zip(features, range(3)):
            scores, _ = predict(weights, active)
            self.assertEqual(scores.index(max(scores)), expected)


if __name__ == '__main__':
    unittest.main()

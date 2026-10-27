"""Export the trained readout and network for the browser simulator."""
import json
import struct
from array import array
from pathlib import Path

from model import Network

ROOT = Path(__file__).resolve().parents[1]


def main():
    network = Network.load(ROOT / 'results/princeton-network.bin')
    manifest = json.loads((ROOT / 'experiments/neuron_groups.json').read_text())
    readout = json.loads((ROOT / 'results/game-supervised-weights.json').read_text())
    visual = json.loads((ROOT / 'experiments/visual_neurons.json').read_text())['neurons']
    groups = manifest['input_groups']
    out = ROOT / 'web-assets'
    out.mkdir(exist_ok=True)
    model = {
        'inputGroups': [[network.index[root] for root in groups['groups'][name]['root_ids']]
                        for name in groups['signal_order']],
        'inputIds': [[str(root) for root in groups['groups'][name]['root_ids']]
                     for name in groups['signal_order']],
        'outputIndices': [network.index[root] for root in readout['input_root_ids']],
        'outputIds': [str(root) for root in readout['input_root_ids']],
        'visualNeurons': [{'id': entry['root_id'], 'node': network.index[int(entry['root_id'])]}
                          for entry in visual],
        'weights': readout['weights'],
        'scale': readout['feature_scale'],
        'weight': readout['settings']['connectome_weight'],
        'steps': readout['settings']['window_ms'],
    }
    (out / 'model.json').write_text(json.dumps(model, separators=(',', ':')) + '\n')
    with (out / 'network.bin').open('wb') as stream:
        stream.write(b'FLYWEB1\n' + struct.pack('<II', len(network.ids), len(network.targets)))
        for values in (network.offsets, network.targets, network.synapses):
            chunk = array('I', values)
            if struct.pack('=I', 1) != struct.pack('<I', 1):
                chunk.byteswap()
            chunk.tofile(stream)
    print(f'Exported {len(network.ids)} neurons and {len(network.targets)} connections.')


if __name__ == '__main__':
    main()

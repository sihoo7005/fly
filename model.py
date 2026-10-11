"""A self-written sparse, discrete-time FlyWire network. Standard library only."""

import argparse
import csv
import json
import math
import struct
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path

from flywire_data import FeatherReader, numbers, read_exact, read_root_ids


class Network:
    def __init__(self, ids, offsets, targets, synapses):
        self.ids = ids
        self.index = {root: i for i, root in enumerate(ids)}
        self.offsets = offsets
        self.targets = targets
        self.synapses = synapses
        n = len(ids)
        if not n or len(self.index) != n or 0 in self.index:
            raise ValueError("Neuron IDs must be nonzero and unique")
        if len(offsets) != n + 1 or offsets[0] != 0 or offsets[-1] != len(targets) or len(targets) != len(synapses):
            raise ValueError("Invalid sparse network layout")
        if any(a > b for a, b in zip(offsets, offsets[1:])) or any(t >= n for t in targets) or any(c < 1 for c in synapses):
            raise ValueError("Invalid edge offset, target or synapse count")

    @classmethod
    def from_batches(cls, ids, batches):
        """Two passes, then merge regions within each source's outgoing edges."""
        index = {root: i for i, root in enumerate(ids)}
        if not ids or len(index) != len(ids) or 0 in index:
            raise ValueError("Invalid neuron list")
        degrees = array("Q", [0]) * len(ids)
        for batch in batches(("pre_pt_root_id",)):
            for root in batch["pre_pt_root_id"]:
                if root not in index:
                    raise ValueError(f"Unknown source neuron: {root}")
                degrees[index[root]] += 1
        offsets = array("Q", [0])
        for degree in degrees:
            offsets.append(offsets[-1] + degree)
        total = offsets[-1]
        targets = array("I", [0]) * total
        synapses = array("Q", [0]) * total
        cursors = array("Q", offsets[:-1])
        for batch in batches(("pre_pt_root_id", "post_pt_root_id", "syn_count")):
            sources = batch["pre_pt_root_id"]
            posts = batch["post_pt_root_id"]
            counts = batch["syn_count"]
            if len(sources) != len(posts) or len(posts) != len(counts):
                raise ValueError("Unequal connection columns")
            for pre, post, count in zip(sources, posts, counts):
                if pre not in index or post not in index or count is None or count < 1:
                    raise ValueError("Invalid connection endpoint or synapse count")
                source = index[pre]
                cursor = cursors[source]
                if cursor >= offsets[source + 1]:
                    raise ValueError("Connection data changed between passes")
                targets[cursor] = index[post]
                synapses[cursor] = count
                cursors[source] += 1
        if any(cursors[i] != offsets[i + 1] for i in range(len(ids))):
            raise ValueError("Connection data changed between passes")
        merged_offsets = array("Q", [0])
        merged_targets = array("I")
        merged_synapses = array("Q")
        for source in range(len(ids)):
            outgoing = {}
            for edge in range(offsets[source], offsets[source + 1]):
                target = targets[edge]
                outgoing[target] = outgoing.get(target, 0) + synapses[edge]
            for target, count in outgoing.items():
                merged_targets.append(target)
                merged_synapses.append(count)
            merged_offsets.append(len(merged_targets))
        return cls(array("Q", ids), merged_offsets, merged_targets, merged_synapses)

    @classmethod
    def from_rows(cls, ids, rows):
        rows = tuple(rows)
        def batches(columns):
            values = {"pre_pt_root_id": [row[0] for row in rows],
                      "post_pt_root_id": [row[1] for row in rows],
                      "syn_count": [row[2] for row in rows]}
            yield {key: values[key] for key in columns}
        return cls.from_batches(ids, batches)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        try:
            with partial.open("wb") as stream:
                stream.write(b"FLYNET1\n" + struct.pack("<QQ", len(self.ids), len(self.targets)))
                for values in (self.ids, self.offsets, self.targets, self.synapses):
                    if sys.byteorder == "little":
                        values.tofile(stream)
                    else:
                        copy = array(values.typecode, values)
                        copy.byteswap()
                        copy.tofile(stream)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with path.open("rb") as stream:
            if read_exact(stream, 8) != b"FLYNET1\n":
                raise ValueError("Unknown network cache format")
            n, edges = struct.unpack("<QQ", read_exact(stream, 16))
            if 24 + n * 8 + (n + 1) * 8 + edges * 12 != path.stat().st_size:
                raise ValueError("Network cache size mismatch")
            ids = numbers(read_exact(stream, n * 8), "Q")
            offsets = numbers(read_exact(stream, (n + 1) * 8), "Q")
            targets = numbers(read_exact(stream, edges * 4), "I")
            synapses = numbers(read_exact(stream, edges * 8), "Q")
        return cls(ids, offsets, targets, synapses)


@dataclass(frozen=True)
class Parameters:
    weight: float = 0.01
    dt_ms: float = 1.0
    tau_ms: float = 20.0
    threshold: float = 1.0
    refractory_ms: float = 2.0

    def __post_init__(self):
        for name in ("weight", "dt_ms", "tau_ms", "threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.refractory_ms) or self.refractory_ms < 0:
            raise ValueError("refractory_ms must be finite and nonnegative")


class Simulator:
    def __init__(self, network, parameters=Parameters(), inhibitory=()):
        self.network = network
        self.parameters = parameters
        self.signs = array("b", [1]) * len(network.ids)
        for root in inhibitory:
            if root not in network.index:
                raise ValueError(f"Unknown inhibitory neuron: {root}")
            self.signs[network.index[root]] = -1
        self.voltage = array("d", [0]) * len(network.ids)
        self.updated = array("Q", [0]) * len(network.ids)
        self.blocked_until = array("Q", [0]) * len(network.ids)
        self.pending = {}
        self.time_step = 0
        self.refractory_steps = math.ceil(parameters.refractory_ms / parameters.dt_ms)

    def step(self, inputs=None):
        """Inputs use root IDs; returned root IDs fired at this step's start time."""
        inputs = inputs or {}
        for root, amount in inputs.items():
            if root not in self.network.index or not math.isfinite(amount):
                raise ValueError("Unknown input neuron or nonfinite input")
        incoming = self.pending
        self.pending = {}
        for root, amount in inputs.items():
            node = self.network.index[root]
            incoming[node] = incoming.get(node, 0.0) + amount
        fired = []
        p = self.parameters
        for node, amount in incoming.items():
            elapsed = (self.time_step - self.updated[node]) * p.dt_ms
            value = self.voltage[node] * math.exp(-elapsed / p.tau_ms) + amount
            self.updated[node] = self.time_step
            if self.time_step < self.blocked_until[node]:
                self.voltage[node] = 0
                continue
            if not math.isfinite(value):
                raise ValueError("Activity overflow; reduce weight or input")
            if value >= p.threshold:
                fired.append(node)
                self.voltage[node] = 0
                self.blocked_until[node] = self.time_step + self.refractory_steps
            else:
                self.voltage[node] = value
        net = self.network
        for source in fired:
            coefficient = p.weight * self.signs[source]
            for edge in range(net.offsets[source], net.offsets[source + 1]):
                target = net.targets[edge]
                self.pending[target] = self.pending.get(target, 0.0) + coefficient * net.synapses[edge]
        self.time_step += 1
        return [net.ids[node] for node in fired]

    def activity(self, root):
        node = self.network.index[root]
        last_step = max(0, self.time_step - 1)
        elapsed = (last_step - self.updated[node]) * self.parameters.dt_ms
        return self.voltage[node] * math.exp(-elapsed / self.parameters.tau_ms)


def run_to_csv(network, parameters, inputs, duration_ms, pulse_every_ms, output, inhibitory=()):
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("Duration must be finite and positive")
    pulse_steps = pulse_every_ms / parameters.dt_ms
    if not math.isfinite(pulse_steps) or pulse_steps < 1 or not math.isclose(pulse_steps, round(pulse_steps)):
        raise ValueError("Pulse interval must be a positive multiple of dt_ms")
    if not inputs or any(root not in network.index for root in inputs):
        raise ValueError("Choose at least one known input neuron")
    simulator = Simulator(network, parameters, inhibitory)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    active = set()
    spikes = 0
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_ms", "root_id"))
        for step in range(math.ceil(duration_ms / parameters.dt_ms)):
            stimulus = {root: parameters.threshold for root in inputs} if step % round(pulse_steps) == 0 else {}
            fired = simulator.step(stimulus)
            for root in fired:
                writer.writerow((f"{step * parameters.dt_ms:.9g}", root))
            active.update(fired)
            spikes += len(fired)
    summary = {"neurons": len(network.ids), "connections": len(network.targets),
               "input_ids": list(inputs), "spikes": spikes, "active_neurons": len(active),
               "duration_ms": duration_ms, "spikes_csv": str(output),
               "weight_per_synapse": parameters.weight,
               "all_excitatory_assumption": not bool(inhibitory)}
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Read real data and build a compact network cache")
    build.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent / "data/raw")
    build.add_argument("--out", type=Path, default=Path("data/network.bin"))
    run = sub.add_parser("run", help="Simulate the cached FlyWire network")
    run.add_argument("--network", type=Path, default=Path("data/network.bin"))
    demo = sub.add_parser("demo", help="Run a four-neuron demonstration without data downloads")
    for command in (run, demo):
        command.add_argument("--input", type=int, nargs="+", help="Input neuron root IDs")
        command.add_argument("--inhibitory", type=int, nargs="*", default=[])
        command.add_argument("--duration", type=float, default=20.0)
        command.add_argument("--pulse-every", type=float, default=10.0)
        command.add_argument("--weight", type=float, default=0.01)
        command.add_argument("--dt", type=float, default=1.0)
        command.add_argument("--tau", type=float, default=20.0)
        command.add_argument("--threshold", type=float, default=1.0)
        command.add_argument("--refractory", type=float, default=2.0)
        command.add_argument("--out", type=Path, default=Path("results/spikes.csv"))
    args = parser.parse_args()
    if args.command == "build":
        ids = read_root_ids(args.folder / "proofread_root_ids_783.npy")
        reader = FeatherReader(args.folder / "proofread_connections_783.feather")
        print("Building from real FlyWire connections (two streaming passes)...", flush=True)
        network = Network.from_batches(ids, reader.batches)
        network.save(args.out)
        print(json.dumps({"neurons": len(ids), "connections": len(network.targets),
                          "synapses": sum(network.synapses), "cache": str(args.out)}), flush=True)
        return
    network = (Network.from_rows([1, 2, 3, 4], [(1, 2, 60), (1, 2, 40), (2, 3, 100), (3, 4, 100)])
               if args.command == "demo" else Network.load(args.network))
    inputs = args.input or [network.ids[max(range(len(network.ids)),
                 key=lambda node: network.offsets[node + 1] - network.offsets[node])]]
    if args.input is None:
        print(f"No input specified; using neuron {inputs[0]} with most outgoing connections.", file=sys.stderr)
    parameters = Parameters(args.weight, args.dt, args.tau, args.threshold, args.refractory)
    run_to_csv(network, parameters, inputs, args.duration, args.pulse_every, args.out, args.inhibitory)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))

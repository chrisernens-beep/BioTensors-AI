"""Quick end-to-end demo: train a small brain and print a decision."""

import pickle
import time

from brain_architecture import BioBrain
from compile_biobrain import BioCompiler


def main():
    print("=== BioTensors AI — quick demo ===\n")

    brain = BioBrain()
    compiler = BioCompiler(brain)

    print("Training core concepts (short run)...")
    start = time.time()
    compiler.train_concept("SOI", iterations=200)
    compiler.train_concept("STABILITE", iterations=200, target_action=0)
    compiler.imprint_ethics(iterations=400)
    elapsed = time.time() - start
    print(f"Training done in {elapsed:.2f}s\n")

    pattern = compiler.library["STABILITE"]
    brain.forward(pattern)
    decision = brain.executive.get_decision()
    print(f"Input concept : STABILITE")
    print(f"Decision      : {decision}")

    with open("demo_brain.bio", "wb") as f:
        pickle.dump(
            {
                "weights_s": brain.sensory.weights,
                "weights_a": brain.associative.weights,
                "weights_e": brain.executive.weights,
                "weights_l": brain.limbic.weights,
                "vocabulary": compiler.library,
            },
            f,
        )
    print("\nSaved weights to demo_brain.bio (gitignored by default).")


if __name__ == "__main__":
    main()

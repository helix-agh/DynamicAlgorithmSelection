"""Entry point dispatcher.

Use the dedicated scripts instead:
    python train.py <name> [options]      — train a PPO agent
    python evaluate.py <name> [options]  — evaluate a saved model
"""

if __name__ == "__main__":
    import sys

    print("Use train.py or evaluate.py directly:")
    print("  python train.py --help")
    print("  python evaluate.py --help")
    sys.exit(0)

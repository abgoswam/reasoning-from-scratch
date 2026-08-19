# Concept: the RLVR training set (notebook cells 38-40)
#
# MATH minus the 500 examples used for evaluation in ch03, so training never
# sees the test set. Only the "problem" and "answer" fields are used.

from pprint import pprint

from trials_00 import load_math_train


def main():
    math_train = load_math_train()

    print("Dataset size:", len(math_train))
    print()
    pprint(math_train[4])
    print()
    print("Fields used by GRPO:", sorted({"problem", "answer"}))


if __name__ == "__main__":
    main()

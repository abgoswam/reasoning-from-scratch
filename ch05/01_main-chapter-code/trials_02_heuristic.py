# Concept: rule-based response scoring (notebook cells 23-27)   [loads model]
#
# The score rewards answer format, not mathematical correctness. Its brevity
# term counts characters rather than tokens.

from trials_00 import heuristic_score, load_base_model, plot_brevity_curve
from trials_01_baseline import generate_responses


def main():
    model, tokenizer, device = load_base_model()
    print("Device:", device)
    response_1, response_2 = generate_responses(model, tokenizer, device)
    plot_brevity_curve(500)
    print(round(heuristic_score(response_1), 3))
    print(round(heuristic_score(response_2), 3))


if __name__ == "__main__":
    main()
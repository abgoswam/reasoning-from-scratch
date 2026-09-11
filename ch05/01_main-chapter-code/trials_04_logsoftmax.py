# Concept: logits, probabilities, and log-probabilities (notebook cells 49-53)
#
# log_softmax matches log(softmax) while keeping the computation numerically
# stable. This synthetic example does not need the model or GPU.

import matplotlib.pyplot as plt
import torch


def main():
    torch.set_printoptions(precision=4, sci_mode=False)
    logits = torch.linspace(-2, 2, steps=7)
    probas = torch.softmax(logits, dim=-1)
    log_probas = torch.log_softmax(logits, dim=-1)
    print(probas)
    print(torch.log(probas))
    print(log_probas)

    plt.figure(figsize=(9, 4))
    for index, (values, title, ylabel) in enumerate((
        (logits, "Logits", "Value"),
        (probas, "torch.softmax(logits)", "Probability"),
        (log_probas, "torch.log_softmax(logits)", "Log-probability"),
    ), start=1):
        plt.subplot(1, 3, index)
        plt.bar(range(len(values)), values, color=f"C{index - 1}", alpha=0.7)
        plt.title(title)
        plt.xlabel("Token index")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("logits_softmax_log_softmax.pdf")
    plt.show()


if __name__ == "__main__":
    main()
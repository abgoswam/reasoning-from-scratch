# Concept: answer confidence scoring (notebook cells 66-74)   [loads model]
#
# Only answer tokens are scored: the first answer token is predicted at the
# final prompt position. Averaging avoids directly penalizing longer answers.

from trials_00 import avg_logprob_answer, calc_next_token_logprobas, load_base_model


def main():
    model, tokenizer, device = load_base_model()
    print("Device:", device)
    prompt = "What is the capital of Germany?"
    answer = " The capital of Germany is Berlin."

    token_logprobs, total = calc_next_token_logprobas(
        model, tokenizer, prompt + answer, device, show=False
    )
    print("Next-token logprobas:", token_logprobs)
    print("Joint log-probability:", total)
    print("Answer tokens:", len(tokenizer.encode(answer)))
    print("Last answer-token scores:", token_logprobs[-7:])
    print("Mean:", token_logprobs[-7:].mean())

    for candidate in (answer, " The capital of Germany is Bridge."):
        score = avg_logprob_answer(
            model, tokenizer, prompt, candidate, device
        )
        print(f"{candidate!r}: {score}")


if __name__ == "__main__":
    main()
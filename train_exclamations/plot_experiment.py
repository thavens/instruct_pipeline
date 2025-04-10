import matplotlib.pyplot as plt

ratios = []
with open("test_sft.log") as f:
    while line := f.readline():
        if "Ratio" in line:
            num = float(line.removeprefix("Ratio: "))
            ratios.append(num)

grpo_ratios = [ratios[-1]]
with open("test_grpo.log") as f:
    while line := f.readline():
        if "Ratio" in line:
            num = float(line.removeprefix("Ratio: "))
            grpo_ratios.append(num)

ckpt_grpo = list(range(450, 450 + 100 * len(grpo_ratios), 100))

ckpt = [50 * i + 50 for i in range(len(ratios))]

sft_line = [[0, 450], [0.5, 0.5]]

plt.plot(ckpt, ratios)
plt.plot(ckpt_grpo, grpo_ratios)
plt.plot(*sft_line, linestyle=":")

plt.legend(["SFT Ratio", "RL Ratio", "GT Ratio"])

plt.savefig("experiment_graph.png")
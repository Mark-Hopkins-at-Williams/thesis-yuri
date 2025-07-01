import json
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("experiments/")
PREFIX = "exp1"
METRIC = "bleu"

def mean(ls):
    return sum(ls) / len(ls)

def read_scores(experiment_dir):
    print(experiment_dir)
    try:
        with open(Path(experiment_dir) / "scores.json") as reader:
            data = json.load(reader)
    except:
        data = None
    return data

x = [512, 1024, 2048, 4096, 8192, 16834]
ys = dict()
for tuning in ['multi', 'bi1', 'bi2']:
    ys[tuning] = []
    for num_train_lines in x:
        exp_dirs = list(BASE_DIR.glob(f"{PREFIX}-{tuning}-{num_train_lines}-v*")) 
        trial_scores = []
        for exp_dir in exp_dirs:
            scores = read_scores(exp_dir)
            if scores is not None:
                score_avg = mean([scores[lang_pair][METRIC] for lang_pair in scores])
                trial_scores.append(score_avg)
        ys[tuning].append(mean(trial_scores))

y_bi = [(y1+y2)/2 for (y1, y2) in zip(ys['bi1'], ys['bi2'])]

plt.plot(x, y_bi, label='bi', color='blue', linestyle='-')
plt.plot(x, ys['multi'], label='multi', color='red', linestyle='--')
plt.xscale('log')
plt.xlabel('num train')
plt.ylabel(METRIC)
plt.title('Experiment')
plt.legend()
plt.savefig(f'{PREFIX}.{METRIC}.png')
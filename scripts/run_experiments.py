"""
scripts/run_experiments.py - the full experimental protocol in one resumable command.

Stage 1-3  Parameter tuning, one aspect at a time, on the VALIDATION classes only
           (training seed 0, 5-shot, selection metric = mean validation mIoU over
           3 evaluation seeds x 50 episodes). Each stage keeps the values chosen
           by the stages before it.
             1. learning rate      {3e-5, 1e-4, 3e-4}   (both methods)
             2. training epochs    {10, 20}              (both methods)
             3a. baseline test-time adaptation steps {0, 1, 5, 10}
             3b. baseline test-time adaptation lr    {3e-5, 1e-4, 3e-4}
             3c. prototype distance {euclidean/sqrt(C), cosine x 20 (PANet)}
Stage 4    Final training with the chosen settings over N training seeds for:
           fine-tuning baseline, prototype method, boundary-weighted prototype.
Stage 5    Test evaluation of every final model at k=1 and k=5 (E evaluation seeds
           x 50 episodes, identical episodes for every model), plus the baseline
           without test-time adaptation (0 steps) as a diagnostic.
Stage 6    Per-class analysis and qualitative examples for the seed-0 models.
Stage 7    experiments/summary.json with every table the report needs.

Everything is written under experiments/ (checkpoints are git-ignored; the small
JSON logs are meant to be committed). Finished runs are skipped, so the command
can simply be re-run after a crash or reboot.

Usage (from the repo root):
    python -m scripts.run_experiments --data-root data/fewshot_data --smoke   # ~5 min check
    python -m scripts.run_experiments --data-root data/fewshot_data           # full run
    python -m scripts.run_experiments --data-root data/fewshot_data --n-seeds 5 --stage tune
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

LRS = [3e-5, 1e-4, 3e-4]
EPOCH_OPTIONS = [10, 20]
TTA_STEPS = [0, 1, 5, 10]
TTA_LRS = [3e-5, 1e-4, 3e-4]
DISTANCES = ["euclidean", "cosine"]
DEFAULT_TTA = (5, 1e-4)
K_TRAIN = 5


class Runner:
    def __init__(self, args):
        self.a = args
        self.root = Path("experiments_smoke" if args.smoke else "experiments")
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        if args.smoke:
            self.epoch_options = [1, 2]
            self.train_extra = ["--episodes-per-epoch", "4", "--val-episodes", "2"]
            self.val_seeds, self.val_eps = [0], 2
            self.test_seeds, self.test_eps = [0, 1], 2
            self.n_seeds = 2
            self.analyze_extra = ["--episodes-per-class", "1", "--n-qualitative", "2", "--max-classes", "5"]
        else:
            self.epoch_options = EPOCH_OPTIONS
            self.train_extra = []
            self.val_seeds, self.val_eps = [0, 1, 2], 50
            self.test_seeds, self.test_eps = list(range(args.eval_seeds)), 50
            self.n_seeds = args.n_seeds
            self.analyze_extra = []
        self.log_path = self.root / "log.txt"

    # ------------------------------------------------------------ utilities
    def log(self, msg):
        line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save_state(self):
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def run(self, cmd, logfile):
        """Run a child process, streaming its output to the console and a log file."""
        self.log("RUN " + " ".join(cmd))
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        with open(logfile, "w", encoding="utf-8") as lf:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", env=env)
            for line in p.stdout:
                sys.stdout.write(line)
                lf.write(line)
            p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"command failed ({p.returncode}); see {logfile}")

    def common(self):
        return ["--data-root", self.a.data_root, "--splits-file", self.a.splits_file]

    # ------------------------------------------------------------ building blocks
    @staticmethod
    def run_name(method, lr, epochs, seed, weighted=False, distance="euclidean", tta=DEFAULT_TTA):
        name = method + ("_weighted" if weighted else "")
        if method == "prototype":
            name += f"_{distance}"
        else:
            name += f"_tta{tta[0]}-{tta[1]:g}"
        return f"{name}_lr{lr:g}_e{epochs}_s{seed}"

    def train(self, method, lr, epochs, seed, weighted=False, distance="euclidean", tta=DEFAULT_TTA):
        name = self.run_name(method, lr, epochs, seed, weighted, distance, tta)
        d = self.root / "train" / name
        h = d / "history.json"
        if h.exists() and json.loads(h.read_text()).get("complete") and (d / "best.pt").exists():
            return d
        d.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "src.train", "--method", method, *self.common(),
               "--k-shot", str(K_TRAIN), "--epochs", str(epochs), "--lr", str(lr), "--seed", str(seed),
               "--out-dir", str(self.root / "train"), "--run-name", name,
               "--adapt-steps", str(tta[0]), "--adapt-lr", str(tta[1]), *self.train_extra]
        if weighted:
            cmd.append("--weighted")
        if method == "prototype":
            cmd += ["--distance", distance]
        self.run(cmd, d / "train_log.txt")
        return d

    def evaluate(self, out, split, shots, seeds, episodes, baseline=None, prototype=None, tta=DEFAULT_TTA):
        out = Path(out)
        if out.exists():
            return json.loads(out.read_text())
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "src.evaluate", *self.common(), "--split", split,
               "--shots", *map(str, shots), "--seeds", *map(str, seeds),
               "--episodes-per-seed", str(episodes), "--out", str(out),
               "--adapt-steps", str(tta[0]), "--adapt-lr", str(tta[1])]
        if baseline:
            cmd += ["--baseline-ckpt", str(Path(baseline) / "best.pt")]
        if prototype:
            cmd += ["--prototype-ckpt", str(Path(prototype) / "best.pt")]
        self.run(cmd, out.with_suffix(".log.txt"))
        return json.loads(out.read_text())

    def val_score(self, run_dir, method, tta=DEFAULT_TTA, tag=""):
        out = self.root / "val" / f"{Path(run_dir).name}{tag}.json"
        kw = {"baseline": run_dir} if method == "baseline" else {"prototype": run_dir}
        res = self.evaluate(out, "val", [K_TRAIN], self.val_seeds, self.val_eps, tta=tta, **kw)
        r = res[f"k{K_TRAIN}_{method}"]
        return {"val_mIoU": r["mIoU_mean"], "val_mIoU_std": r["mIoU_std"],
                "val_F1": r["F1_mean"], "run": Path(run_dir).name}

    def choose(self, key, table, method):
        best = max(table, key=lambda row: row["val_mIoU"])
        self.state.setdefault("tuning", {}).setdefault(method, {})[key] = {"table": table, "chosen": best["value"]}
        self.save_state()
        self.log(f"TUNED {method} {key}: " + ", ".join(f"{r['value']}={r['val_mIoU']:.4f}" for r in table)
                 + f"  -> chosen {best['value']}")
        return best["value"]

    # ------------------------------------------------------------ stages
    def tune(self):
        chosen = {}
        for method in ["baseline", "prototype"]:
            # Stage 1: learning rate (10 epochs, default TTA / euclidean)
            e0 = self.epoch_options[0]
            table = []
            for lr in LRS:
                d = self.train(method, lr, e0, 0)
                table.append({"value": lr, **self.val_score(d, method)})
            lr = self.choose("learning_rate", table, method)
            # Stage 2: epochs
            table = []
            for ep in self.epoch_options:
                d = self.train(method, lr, ep, 0)
                table.append({"value": ep, **self.val_score(d, method)})
            epochs = self.choose("epochs", table, method)
            chosen[method] = {"lr": lr, "epochs": epochs}

        # Stage 3a/3b: baseline test-time adaptation (evaluation only, no retraining)
        b = chosen["baseline"]
        bdir = self.train("baseline", b["lr"], b["epochs"], 0)
        table = []
        for steps in TTA_STEPS:
            tta = (steps, DEFAULT_TTA[1])
            table.append({"value": steps, **self.val_score(bdir, "baseline", tta, tag=f"_eval-tta{steps}-{tta[1]:g}")})
        steps = self.choose("tta_steps", table, "baseline")
        tta_lr = DEFAULT_TTA[1]
        if steps > 0:
            table = []
            for tlr in TTA_LRS:
                tta = (steps, tlr)
                table.append({"value": tlr, **self.val_score(bdir, "baseline", tta, tag=f"_eval-tta{steps}-{tlr:g}")})
            tta_lr = self.choose("tta_lr", table, "baseline")
        chosen["baseline"]["tta"] = [steps, tta_lr]

        # Stage 3c: prototype distance (needs one extra training run)
        p = chosen["prototype"]
        table = []
        for dist in DISTANCES:
            d = self.train("prototype", p["lr"], p["epochs"], 0, distance=dist)
            table.append({"value": dist, **self.val_score(d, "prototype")})
        chosen["prototype"]["distance"] = self.choose("distance", table, "prototype")

        self.state["chosen"] = chosen
        self.save_state()
        self.log("CHOSEN SETTINGS: " + json.dumps(chosen))

    def final(self):
        if "chosen" not in self.state:
            raise SystemExit("Run the tuning stage first (--stage tune or --stage all).")
        c = self.state["chosen"]
        b, p = c["baseline"], c["prototype"]
        tta = tuple(b["tta"])
        conds = {
            "baseline": lambda s: self.train("baseline", b["lr"], b["epochs"], s, tta=tta),
            "prototype": lambda s: self.train("prototype", p["lr"], p["epochs"], s, distance=p["distance"]),
            "prototype_weighted": lambda s: self.train("prototype", p["lr"], p["epochs"], s,
                                                       weighted=True, distance=p["distance"]),
        }
        final_runs = {name: [] for name in conds}
        for s in range(self.n_seeds):          # seed-major, so partial results are balanced
            for name, fn in conds.items():
                final_runs[name].append(str(fn(s)))
        self.state["final_runs"] = final_runs
        self.save_state()

        # Stage 5: test evaluation (identical episodes for every model)
        shots = [1, 5]
        for s in range(self.n_seeds):
            self.evaluate(self.root / "test" / f"s{s}_baseline_prototype.json", "test", shots, self.test_seeds,
                          self.test_eps, baseline=final_runs["baseline"][s],
                          prototype=final_runs["prototype"][s], tta=tta)
            self.evaluate(self.root / "test" / f"s{s}_weighted.json", "test", shots, self.test_seeds,
                          self.test_eps, prototype=final_runs["prototype_weighted"][s])
            self.evaluate(self.root / "test" / f"s{s}_baseline_noTTA.json", "test", shots, self.test_seeds,
                          self.test_eps, baseline=final_runs["baseline"][s], tta=(0, tta[1]))

        # Stage 6: per-class + qualitative for the seed-0 models (same episodes)
        for name, method in [("baseline", "baseline"), ("prototype", "prototype")]:
            out_dir = self.root / "analysis" / f"{name}_k5"
            if (out_dir / "per_class_results.json").exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "scripts.analyze", *self.common(),
                   "--checkpoint", str(Path(final_runs[name][0]) / "best.pt"), "--method", method,
                   "--k-shot", "5", "--out-dir", str(out_dir), *self.analyze_extra]
            if method == "baseline":
                cmd += ["--adapt-steps", str(tta[0]), "--adapt-lr", str(tta[1])]
            self.run(cmd, out_dir / "analyze_log.txt")

    def summarise(self):
        """Collect everything the report needs into experiments/summary.json."""
        st = self.state
        summary = {"tuning": st.get("tuning"), "chosen": st.get("chosen"), "final": {}, "cost": {}}
        runs = st.get("final_runs", {})
        # per-training-seed test scores (mean over the evaluation seeds)
        rows = {}
        for s in range(len(runs.get("baseline", []))):
            for fname, keys in [(f"s{s}_baseline_prototype.json", [("baseline", "baseline"), ("prototype", "prototype")]),
                                (f"s{s}_weighted.json", [("prototype", "prototype_weighted")]),
                                (f"s{s}_baseline_noTTA.json", [("baseline", "baseline_noTTA")])]:
                f = self.root / "test" / fname
                if not f.exists():
                    continue
                res = json.loads(f.read_text())
                for k in (1, 5):
                    for label, cond in keys:
                        r = res.get(f"k{k}_{label}")
                        if r:
                            rows.setdefault(f"k{k}_{cond}", []).append(
                                {"train_seed": s, "mIoU": r["mIoU_mean"], "F1": r["F1_mean"],
                                 "eval_seed_mIoU_std": r["mIoU_std"], "ms_per_episode": r["ms_per_episode"],
                                 "per_seed_mIoU": r["per_seed_mIoU"], "per_seed_F1": r["per_seed_F1"]})
        summary["final"] = rows
        # training cost from the history files
        for cond, dirs in runs.items():
            dirs = [Path(str(d).replace("\\", "/")) for d in dirs]  # runs from Windows store "\\" paths
            hs = [json.loads((d / "history.json").read_text()) for d in dirs if (d / "history.json").exists()]
            if hs:
                summary["cost"][cond] = {
                    "train_time_s": [h["total_train_time_s"] for h in hs],
                    "best_epoch": [h["best_epoch"] for h in hs],
                    "peak_gpu_mem_MB": [h.get("peak_gpu_mem_MB") for h in hs],
                    "total_params": hs[0]["total_params"],
                    "trainable_params": hs[0]["trainable_params"],
                    "device": hs[0].get("device"),
                }
        out = self.root / "summary.json"
        out.write_text(json.dumps(summary, indent=2))
        self.log(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits-file", default="configs/class_splits.json")
    ap.add_argument("--stage", choices=["all", "tune", "final"], default="all")
    ap.add_argument("--n-seeds", type=int, default=10, help="training seeds per final condition")
    ap.add_argument("--eval-seeds", type=int, default=10, help="evaluation seeds per final model")
    ap.add_argument("--smoke", action="store_true", help="tiny version of everything, to check the pipeline")
    args = ap.parse_args()
    r = Runner(args)
    r.log(f"=== start: stage={args.stage} smoke={args.smoke} n_seeds={r.n_seeds} ===")
    if args.stage in ("all", "tune"):
        r.tune()
    if args.stage in ("all", "final"):
        r.final()
    r.summarise()
    r.log("=== done ===")


if __name__ == "__main__":
    main()

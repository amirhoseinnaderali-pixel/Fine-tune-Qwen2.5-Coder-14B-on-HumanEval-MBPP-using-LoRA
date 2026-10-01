import argparse
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        data = json.loads(open(path, encoding="utf-8").read())
        rows.append(
            {
                "mode": data["mode"],
                "examples": data["examples"],
                "task_pass_rate": data["task_pass_rate"],
                "test_pass_rate": data["test_pass_rate"],
                "challenge_task_rate": data["challenge_task_rate"],
            }
        )
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    frame.to_csv("results/summary.csv", index=False)


if __name__ == "__main__":
    main()

import argparse
import pathlib

import matplotlib.pyplot as plt

from milptune.version import VERSION


def distances(actual, estimated, output_file):
    plt.clf()
    plt.scatter(estimated, actual)
    plt.plot(range(0, 100), range(0, 100), color="silver", linestyle="dashed")

    plt.xlabel("Nearest Neighbor Cost", fontsize=15)
    plt.ylabel("Validation Instance Cost", fontsize=15)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.grid()
    plt.savefig(output_file, bbox_inches="tight")


class CapitalisedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if not prefix:
            prefix = "Usage: "
            return super(CapitalisedHelpFormatter, self).add_usage(usage, actions, groups, prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=CapitalisedHelpFormatter,
        description="Suggests configuration parameters for SCIP",
    )
    parser._positionals.title = "Positional arguments"
    parser._optionals.title = "Optional arguments"
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"MILPTune v{VERSION}",
        help="Shows program's version number and exit",
    )
    parser.add_argument(
        "validation_dir", type=str, help="Specifies the validation dir that has .csv files"
    )
    args = parser.parse_args()

    validation_path = pathlib.Path(args.validation_dir)
    instances = list(validation_path.glob("*.csv"))

    actual = []
    estimated = []
    for instance in instances:
        with open(instance, "r") as f:
            a, e = [], []
            for line in f:
                source, rank, estimated_cost, actual_cost, time, _, distance = line.strip().split(
                    ";"
                )
                if source == "milptune":
                    a.append(float(actual_cost))
                    e.append(float(estimated_cost))
            actual.append(min(a))
            estimated.append(min(e))

    distances(actual, estimated, "distance.pdf")

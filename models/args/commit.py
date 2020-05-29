#!/usr/bin/env python

import sys


def check_models(commit_file):
    """
    """
    models = set()
    commit = []
    with open(commit_file, 'r') as fin:
        commit = fin.readlines()
        commit.reverse()
    for item in commit[1:]:
        line = item.strip()
        if line == "":
            break
        get_models(line.split()[0], models)
    return models


def get_models(line, models):
    """
    """
    # check line
    model = line
    models.add(model)


if __name__ == "__main__":
    commit_file = sys.argv[1]
    check_models(commit_file)

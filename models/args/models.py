#!/usr/bin/env python

import sys
import os


def check_models(models_file, register_file):
    """
    """
    models = set()
    register = []
    with open(register_file, 'r') as fin:
        for line in fin:
            register.append(line.strip())
            models.add(line.strip().split()[0])

    #print(models)
    #print(register)
    commit = set()
    with open(models_file, 'r') as fin:
        for line in fin:
            tmp = line.strip()
            for item in models:
                if item in tmp:
                    commit.add(item)
    #print(commit)
    test_case = []
    for item in register:
        if item.split()[0] in commit:
            test_case.append(item)
            print(item)

    #print(test_case)
    return test_case


if __name__ == "__main__":
    models_file = sys.argv[1]
    register_file = sys.argv[2]
    check_models(models_file, register_file)

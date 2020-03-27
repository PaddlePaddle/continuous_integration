# encoding:utf-8

import json
import os
import sys


def main(commitid, directory, result_output):
    ce_path = "../ce/{}/{}".format(commitid, directory)

    # run case
    os.system("cd ../bin && python run_ce.py fast {}/{}".format(commitid, directory))

    # merge result
    all_results = []
    index = 1
    for _file in sorted(os.listdir(os.path.join(ce_path, "result"))):
        if _file.endswith(".result"):
            result_json = {"id": index, "op": _file[:-7]}
            with open(os.path.join(ce_path, "result", _file), "r") as fr:
                file_json = json.load(fr)
            if file_json["function"].startswith("P"):
                result_json["paddle_vs_tf"] = 1
            elif file_json["function"].startswith("F"):
                result_json["paddle_vs_tf"] = -1
            else:
                result_json["paddle_vs_tf"] = 0
            result_json["consistency"] = 1 if file_json["stablity"].startswith("P") else -1
            with open(os.path.join(ce_path, "log", _file.replace(".result", ".fast")), "r") as fr:
                result_json["console"] = fr.read()
            index += 1
            all_results.append(result_json)

    with open(result_output, "w") as fw:
        json.dump(all_results, fw)


if __name__ == "__main__":
    commitid = sys.argv[1]
    directory = sys.argv[2]
    result_output = sys.argv[3]
    main(commitid, directory, result_output)

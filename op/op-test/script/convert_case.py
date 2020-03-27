# coding:utf-8

import os
import sys
import json
import subprocess


op_key_dict = {
    #"top_k": "topk"
}

def main():
    cmd = 'nvidia-docker exec mysql ./mysql -e "select op, is_gradient, tf, level_diff, \`return\` from paddle.op_meta;"'
    sp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = sp.communicate()
    op_meta = {}
    for line in stdout.strip().split("\n")[1:]:
        items = line.split("\t")
        op = items[0]
        is_gradient = int(items[1])
        tf = items[2]
        level_diff = int(items[3]) if items[3] != "0" else None
        return_info = eval(items[4]) if items[4] != "NULL" else None
        op_meta[op] = {
            "is_gradient": is_gradient,
            "tf": tf,
            "level_diff": level_diff,
            "return_info": return_info
        }

    cmd = 'nvidia-docker exec mysql ./mysql -e "select op, \`key\`, \`value\`, gradient, batch_size, feed from paddle.op_param_meta;"'
    sp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = sp.communicate()
    op_param_meta = {}
    for line in stdout.strip().split("\n")[1:]:
        items = line.split("\t")
        op = items[0]
        key = items[1]
        value = items[2]
        gradient = int(items[3]) if items[3] != "NULL" else None
        batch_size = int(items[4]) if items[4] != "NULL" else None
        feed = items[5] if items[5] != "NULL" else None
        if op in op_param_meta:
            op_param_meta[op][key] = {
                "tf": value,
                "gradient": gradient,
                "batch_size": batch_size,
                "feed": feed
            }
        else:
            op_param_meta[op] = {key: {
                "tf": value,
                "gradient": gradient,
                "batch_size": batch_size,
                "feed": feed
            }}

    cmd = 'nvidia-docker exec mysql ./mysql -e "select case_name, op, param_info, model from paddle.case_from_model where update_time = (select max(update_time) from paddle.case_from_model);"'
    sp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = sp.communicate()
    for line in stdout.strip().split("\n")[1:]:
	print line
        items = line.split("\t")
        case_key = items[0]
        op = items[1]
        if op in op_key_dict:
            op = op_key_dict[op]
        param_info = items[2].lower()
        model = items[3]

        params = []
        gradient = []
        for param in param_info.replace("\\n", "\n").strip().split("\n"):
            try:
                param_name, param_attr = param.split("--")
                param_name = param_name.strip()
                param_attr = param_attr.strip()
            except Exception as e:
                print line
                continue
            tf_param_name = None
            if op in op_param_meta and param_name in op_param_meta[op]:
                tf_param_name = op_param_meta[op][param_name]["tf"]
            batch_size = 16 # default
            if op in op_param_meta and param_name in op_param_meta[op]:
                if op_param_meta[op][param_name]["batch_size"] is not None:
                    batch_size = op_param_meta[op][param_name]["batch_size"]
            attrs = param_attr.strip().split("|")
            param_type = attrs[0].strip()
            if param_type == "variable":
                dtype = attrs[1].strip()
                shape = eval(attrs[2].strip().split("shape:")[1])
                feed = None
                if len(attrs) > 3:
                    feed = attrs[3].strip().split("feed:")[1].strip()
                _param = {
                    "name": param_name,
                    "type": "variable",
                    "dtype": dtype,
                    "dim": len(shape),
                    "data_generator": [ {"type": "default", "value": size} if size != -1 else {"type": "default", "value": batch_size} for size in shape ]
                }
                if tf_param_name is not None:
                    _param["tf-name"] = tf_param_name
                if feed is not None:
                    _param["feed"] = feed
                elif op in op_param_meta and param_name in op_param_meta[op]:
                    if op_param_meta[op][param_name]["feed"] is not None:
                        _param["feed"] = op_param_meta[op][param_name]["feed"]
                    
                params.append(_param)
                if op in op_param_meta and param_name in op_param_meta[op]:
                    if op_param_meta[op][param_name]["gradient"] is not None:
                        gradient.append(op_param_meta[op][param_name]["gradient"])
            elif param_type in ("list", "tuple"):
                value = eval(attrs[1].strip())
                _param = {
                    "name": param_name,
                    "type": param_type,
                    "size": len(value),
                    "data_generator": [ {"type": "default", "value": size} for size in value ]
                }
                if tf_param_name is not None:
                    _param["tf-name"] = tf_param_name
                params.append(_param)
            else:
                if param_type == "int":
                    value = int(attrs[1].strip())
                #fix a bug ,needs to be compatible float32
                elif param_type == "float" or param_type == "float32":
                    value = float(attrs[1].strip())
                elif param_type == "bool":
                    value = "False" if attrs[1].strip() == "false" else "True"
                elif param_type == "string":
                    value = attrs[1].strip().upper() if attrs[1].strip() in ("nchw", "nhwc") else attrs[1].strip()
                else:
                    raise Exception(param_type)
                _param = {"name": param_name, "type": param_type, "data_generator": {"type": "default", "value": value}}
                if tf_param_name is not None:
                    _param["tf-name"] = tf_param_name
                params.append(_param)

        case_json = {}
        for core_type in ("cpu", "gpu"):
            case_name = "[{}]{}_{}".format(core_type, op, case_key)
            case_json[case_name] = {
                "op": op,
                "params": params
            }
            if core_type == "gpu":
                case_json[case_name]["gpu"] = 1
            if op in op_meta:
                case_json[case_name]["tf-op"] = op_meta[op]["tf"]
                if op_meta[op]["return_info"] is not None:
                    case_json[case_name]["return"] = op_meta[op]["return_info"]
                if op_meta[op]["level_diff"] is not None:
                    case_json[case_name]["level-diff"] = op_meta[op]["level_diff"]
                if op_meta[op]["is_gradient"] == 1 and len(gradient) > 0:
                    case_json[case_name]["gradient"] = gradient


        with open(os.path.join("../ce_bak/from_model/case", "{}_{}.json".format(op, case_key)), "w") as f:
            json.dump(case_json, f, indent=4)


if __name__ == "__main__":
    if len(os.listdir("../ce_bak/from_model/case"))!=0:
    	os.system("rm ../ce_bak/from_model/case/*")
    main()

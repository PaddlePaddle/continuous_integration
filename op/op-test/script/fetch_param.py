# coding:utf-8

import sys
import os
import json
import time


def main():
    update_time = int(time.time())
    op_json_list = os.listdir("../ce_bak/from_model/case")
    for op_json in op_json_list:
        with open(os.path.join("../ce_bak/from_model/case", op_json), "r") as fr:
            op_dict = json.load(fr)
        for case_name in op_dict:
            op = op_dict[case_name]["op"]
            is_gradient = 0 if op_dict[case_name].get("gradient", 0) == 0 else 1
            core_type = op_dict[case_name].get("gpu", 0)
            print case_name, op, is_gradient, core_type
            param_list = []
            for param in op_dict[case_name].get("params"):
                if "name" not in param:
                    continue
                param_info = {}
                if param["type"] == "variable":
                    lod_level = str(param.get("lod_level", "-"))
                    lod = str(param.get("lod", "-"))
                    param_info[param["name"]] = ["variable", param.get("dtype", "float32"), "shape:"+str([item["value"] for item in param.get("data_generator")]), "feed:"+str(param.get("feed", "random.randn"))]
                    if lod_level != "-":
                        param_info[param["name"]].append("lod_level:" + str(lod_level))
                        param_info[param["name"]].append("lod:" + str(lod))
                elif param["type"] in ("list", "tuple"):
                    param_info[param["name"]] = [param["type"], str([item["value"] if item["type"] == "default" else "choice:"+str(item["option"]) for item in param.get("data_generator")])]
                    if "is_tensor" in param:
                        param_info[param["name"]].append("has_tensor:"+str(param["is_tensor"]))
                elif param["type"] in ("int", "float", "string", "bool"):
                    param_info[param["name"]] = [param["type"], str(param.get("data_generator").get("value")) if param.get("data_generator").get("type") == "default" else "choice:"+str(param.get("data_generator").get("option"))]
                elif param["type"] == "param_attr":
                    param_info[param["name"]] = [param["type"], str(param.get("attribute"))]
                else:
                    param_info[param["name"]] = [param["type"], str(param)]
                param_list.append(param_info)
            param_str = "\n".join(["\n".join([str(key) + "--" + "|".join(param_info[key]) for key in param_info]) for param_info in param_list])
            cmd = 'nvidia-docker exec mysql ./mysql -e "insert into paddle.case_meta(case_name, op, is_gradient, core_type, param_info, case_source, update_time) values(\'{}\', \'{}\', {}, {}, \\"{}\\", {}, {}) on duplicate key update param_info=values(param_info), op=values(op), is_gradient=values(is_gradient), core_type=values(core_type), case_source=values(case_source), update_time=values(update_time);"'.format(case_name, op, is_gradient, core_type, param_str, 1, update_time)
            os.system(cmd)
            #if op_dict[case_name].get("tf-disable", 0):
            #    cmd = 'nvidia-docker exec mysql ./mysql -e "insert into paddle.case_meta(case_name, tf_performance, tf_performance_forward) values(\'{}\', {}, {}) on duplicate key update tf_performance=values(tf_performance), tf_performance_forward=values(tf_performance_forward);"'.format(case_name, 0, 0)
            #    os.system(cmd)


def detect_op_param_meta():
    op_json_list = os.listdir("../ce_bak/performance/case")
    for op_json in op_json_list:
        op = op_json[:-5]
        with open(os.path.join("../ce_bak/performance/case", op_json), "r") as fr:
            op_dict = json.load(fr)
        tf_info = {"relation": {}, "other": {}}
        relation_flag = True
        gradient = {}
        for case_name in op_dict:
            #print "###", op, case_name, "###"
            if "tf-op" not in op_dict[case_name]:
                continue
            index = 0
            for param in op_dict[case_name].get("params"):
                if "name" not in param or "tf-name" not in param:
                    print "lack one param"
                    relation_flag = False
                    break
                tf_info["relation"][param["name"]] = param["tf-name"]
                param_info = {}
                if param["type"] == "variable":
                    is_gradient = op_dict[case_name].get("gradient", [])
                    if len(is_gradient) > 0:
                        gradient[param["name"]] = is_gradient[index]
                        index += 1
                    lod_level = str(param.get("lod_level", "-"))
                    lod = str(param.get("lod", "-"))
                    param_info[param["name"]] = ["variable", param.get("dtype", "float32"), "shape:"+str([item["value"] for item in param.get("data_generator")]), "feed:"+str(param.get("feed", "random.randn"))]
                    if lod_level != "-":
                        param_info[param["name"]].append("lod_level:" + str(lod_level))
                        param_info[param["name"]].append("lod:" + str(lod))
                elif param["type"] in ("list", "tuple"):
                    param_info[param["name"]] = [param["type"], str([item["value"] if item["type"] == "default" else "choice:"+str(item["option"]) for item in param.get("data_generator")])]
                    if "is_tensor" in param:
                        param_info[param["name"]].append("has_tensor:"+str(param["is_tensor"]))
                elif param["type"] in ("int", "float", "string", "bool"):
                    param_info[param["name"]] = [param["type"], str(param.get("data_generator").get("value")) if param.get("data_generator").get("type") == "default" else "choice:"+str(param.get("data_generator").get("option"))]
                elif param["type"] == "param_attr":
                    param_info[param["name"]] = [param["type"], str(param.get("attribute"))]
                else:
                    param_info[param["name"]] = [param["type"], str(param)]
            for key in op_dict[case_name]:
                if key in ("return", "return_numpy", "level-diff"):
                    tf_info["other"][key] = op_dict[case_name][key]
        if relation_flag:
            for key in tf_info["relation"]:
                print op, key, tf_info["relation"][key], gradient.get(key)
                cmd = 'nvidia-docker exec mysql ./mysql -e "insert into paddle.op_param_meta(op, type, \\`key\\`, \\`value\\`, gradient) values(\'{}\', {}, \'{}\', \'{}\', {}) on duplicate key update value=values(value), gradient=values(gradient);"'.format(op, 0, key, tf_info["relation"][key], gradient.get(key) if gradient.get(key) is not None else "null")
                os.system(cmd)
            #for key in tf_info["other"]:
                #cmd = 'nvidia-docker exec mysql ./mysql -e "insert into paddle.op_param_meta(op, type, \\`key\\`, \\`value\\`) values(\'{}\', {}, \'{}\', \'{}\') on duplicate key update value=values(value);"'.format(op, 1, key, tf_info["other"][key])
                #os.system(cmd)


def detect_op_meta():
    op_json_list = os.listdir("../ce_bak/performance/case")
    for op_json in op_json_list:
        op = op_json[:-5]
        with open(os.path.join("../ce_bak/performance/case", op_json), "r") as fr:
            op_dict = json.load(fr)
        for case_name in op_dict:
            #print "###", op, case_name, "###"
            if "tf-op" not in op_dict[case_name]:
                continue
            gradient = op_dict[case_name].get("gradient", [])
            tf_op = op_dict[case_name].get("tf-op")
            print op, tf_op, gradient
            cmd = 'nvidia-docker exec mysql ./mysql -e "insert into paddle.op_meta(op, tf, is_gradient) values(\'{}\', \'{}\', {}) on duplicate key update tf=values(tf), is_gradient=values(is_gradient);"'.format(op, tf_op, 0 if len(gradient) == 0 else 1)
            os.system(cmd)
            break

if __name__ == "__main__":
    main()
    #detect_op_param_meta()
    #detect_op_meta()

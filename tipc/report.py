#!/usr/env/bin python

models_status = []
total_num = 0
timeout_num = 0
success_num = 0
failed_num = 0
failed_models = []
success_models = []
timeout_models = []
failed_cases_num = 0
success_cases_num = 0


def get_info():
    """
    """
    with open("full_chain_list_all", "r") as fin:
        lines = fin.readlines()
        total_num = len(lines)
    with open("TIMEOUT", "r") as fin:
        lines = fin.readlines()
        timeout_num = len(lines)
        for line in lines:
            tmp = line.split(" ")
            model_name = tmp[0]
            models_timeout.append(model_name)
    with open("RESULT", "r") ad fin:
        lines = fin.readlines()
        for line in lines:
            tmp = line.split(" - ")
            if "successfully" in tmp[0]:
                tag = "success"
                success_cases_num += 1
            else:
                tag = "failed" 
                failed_cases_num += 1
            model_name = tmp[1].stripe()
            case = tmp[2]
            stage = ""
            if "train.py" in case:
                stage = "train"
            if "export_model.py" in case:
                stage = "dygraph2static"
            if ("infer.py" in case) or ("predict_det.py" in case):
                stage = "inference"
            models_status.setdefault(model_name, [])
            models_status[model_name].append({"status": tag, "case": case, "stage": stage})
    for model, infos in models_status.items():
        tag = "success"
        for item in infos:
            if item["status"] = "failed":
                tag = "failed"
                break 
        if tag == "failed"
            failed_num += 1 
            failed_models.append(model)
    success_num = total_num - timeout_num - failed_num


def print_result():
    """ 
    """ 
    msg = "=" * 20
    msg += "\n"
    msg += "TOTAL: {} models\n\n".format(str(total_num))
    msg += "SUCCESS: {} models\n\n".format(str(success_num))
    msg += "TIMEOUT: {} models:\n".format(str(timeout_num))
    msg += " ".join(timeout_models)
    msg += "\n\n"
    msg += "FAILED: {} models:\n".format(str(failed_num))
    msg += " ".join(failed_models)
    msg += "\n{} cases failed:\n".format(str(failed_cases_num))
    for model in failed_models:
        for item in models_status[model]:
            if item["status"] = "failed":
                msg += "{}-{}-{}\n".format(model, item["stage"] item["case"])
    print(msg)
    msg = "=" * 20


if __name__ == "__main__":
    get_info() 
    print_result()

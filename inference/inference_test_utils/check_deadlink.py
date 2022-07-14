import os
import re
from xml.dom.minidom import parse
from urllib import request
import codecs
import markdown
import argparse


def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--changed_files", type=str, default="./changed_files.txt",
                        help="file contains all changed files in current PR")
    return parser.parse_args()


def md2html(file_path):
    input_file = codecs.open(file_path, mode="r", encoding="utf-8")
    text = input_file.read()

    html = markdown.markdown(text)

    output_file = codecs.open(f"{file_path}.html", mode="w", encoding="utf-8")
    output_file.write(html)


def exec_linkchecker(file_path):
    link_checker_cmd = f"linkchecker {file_path}.html --verbose --output=xml --check-extern > {file_path}.xml"

    os.system(link_checker_cmd)


def parse_linkchecker_result(file_path):
    dom = parse(f"{file_path}.xml")
    data = dom.documentElement

    urldatas = data.getElementsByTagName('urldata')

    total_url = []

    for item in urldatas:
        real_url = item.getElementsByTagName('realurl')[0].childNodes[0].data
        if real_url.startswith("http"):
            total_url.append(real_url)

    total_url = list(set(total_url))

    return total_url


def http_link_check(link, file_path):
    status_flag = False
    for _ in range(3):
        try:
            with request.urlopen(link, timeout=10) as res:
                status = res.status
                if res.status in [200, 401, 403, 503]:
                    print('{} check passed'.format(link))
                    status_flag = True
                    status = "[200]"
                    break
        except Exception as e:
            print('{} check failed'.format(link))
            print("Error as follows:")
            status = f"[{e}]"
            print(e)
    else:
        print('{} check failed, The reasons for the failure are as follows:\n{}'.format(link, status))

    result = f"{status}:{file_path}:{link}"

    return status_flag, result


def relative_link_check(file_path):
    file_dir = os.path.dirname(os.path.abspath(file_path))
    dead_links = []
    with open(file_path, "r") as file:
        data = file.read()
    regex = r"\[.*?\]\((.*?)\)"
    link_list = re.findall(regex, data)

    reg_a_label = r'<a name="(.*?)"> *</a>'
    a_label_list = re.findall(reg_a_label, data)

    relative_links = []
    a_label_links = []
    for link in link_list:
        if link.startswith("http") is False:
            if "#" in link:
                a_label_links.append(link)
            else:
                relative_links.append(link)

    relative_files = [f"{file_dir}/{link}" for link in relative_links]
    for i, file in enumerate(relative_files):
        if os.path.exists(file) is False:
            dead_links.append(f"[404 Not Found]:{file_path}:{relative_links[i]}")
        else:
            print(f"{relative_files[i]} check passed")

    for i, link in enumerate(a_label_links):
        file_name, a_label_name = link.split("#")
        if file_name:
            file = f"{file_dir}/{file_name}"
            if os.path.exists(file) is False:
                dead_links.append(f"[404 Not Found]:{file_path}:{a_label_links[i]}")
                a_labels = []
            else:
                with open(f"{file_dir}/{file_name}", "r") as f:
                    a_labels = re.findall(reg_a_label, f.read())
        else:
            a_labels = a_label_list

        if a_label_name not in a_labels:
            dead_links.append(f"[404 Not Found]:{file_path}:{a_label_links[i]}")
        else:
            print(f"{a_label_links[i]} check passed")

    for i in dead_links:
        print(i)
    return dead_links


def main():
    all_dead_links = []
    with open(args.changed_files, "r") as f:
        file_list = [file.strip() for file in f.readlines()]
    os.system("rm -rf dead_links.txt")

    for single_file in file_list:
        if os.path.exists(single_file) is False:
            continue
        if single_file.endswith(".md") or single_file.endswith(".rst"):
            md2html(single_file)
            print(single_file)
            exec_linkchecker(single_file)
            all_urls = parse_linkchecker_result(single_file)
            for link in all_urls:
                flag, msg = http_link_check(link, single_file)
                if not flag:
                    all_dead_links.append(msg)
            relative_dead_links = relative_link_check(single_file)
            all_dead_links.extend(relative_dead_links)

    if all_dead_links:
        with open("dead_links.txt", "a") as f:
            for link in all_dead_links:
                f.write(f"{link}\n")
        with open("dead_links.txt", "r") as f:
            print("All dead links:")
            print(f.read())
        exit(8)
    else:
        print("All links check passed")


if __name__ == '__main__':
    args = parse_args()
    main()

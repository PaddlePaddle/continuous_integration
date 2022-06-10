import os
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


def link_check(link, file_path):
    status_flag = False
    for _ in range(3):
        try:
            with request.urlopen(link, timeout=10) as res:
                status = res.status
                if res.status in [200, 401, 403]:
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


def main():
    all_dead_links = []
    with open(args.changed_files, "r") as f:
        file_list = [file.strip() for file in f.readlines()]
    os.system("rm -rf dead_links.txt")

    for single_file in file_list:
        if single_file.endswith(".md") or single_file.endswith(".rst"):
            md2html(single_file)
            print(single_file)
            exec_linkchecker(single_file)
            all_urls = parse_linkchecker_result(single_file)
            for link in all_urls:
                flag, msg = link_check(link, single_file)
                if not flag:
                    all_dead_links.append(msg)

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

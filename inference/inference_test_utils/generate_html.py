# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import logging
import time
import urllib.request as urllib2
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    # task info
    parser.add_argument(
        "--task_name", type=str, default="", help="pipleine task name")
    parser.add_argument("--repo", type=str, default="", help="task repo name")
    parser.add_argument(
        "--branch", type=str, default="", help="task branch name")
    parser.add_argument(
        "--commit_id", type=str, default="", help="task commit id")

    # for local html convert
    parser.add_argument(
        "--html_table_path",
        type=str,
        default="./benchmark.html",
        help="pandas table html")
    parser.add_argument(
        "--html_output_path",
        type=str,
        default="./report.html",
        help="final report html")
    return parser.parse_args()


class HtmlContentRender(object):
    """
    Html Content Render for visualizing model task info
    """

    def __init__(self, save_path, href_url="127.0.0.1"):
        """
        __init__
        Args:
            save_path (str): path of save html
            href_url (str, optional): future task url. Defaults to "127.0.0.1".
        """
        self.save_path = save_path
        self.href_url = href_url

    def save_html(self, path, template):
        """
        save html
        Args:
            path (str): output save html path
            template (str): html content
        """
        with open(path, 'w', encoding='utf-8') as file:
            file.write(template)

    def generate_html(self, task_info):
        """
        Args:
            task_info (dict): information need to render inside html
        Returns:
            html_content (str): 
        """
        current_path = os.path.dirname(os.path.abspath(__file__))
        templateLoader = FileSystemLoader(searchpath=current_path)
        env = Environment(loader=templateLoader)
        template = env.get_template('default_template.html')

        html_content = template.render(
            href_url=self.href_url, task_info=task_info)
        return html_content

    def change_time(self, time_stamp, fmt='%Y-%m-%d %H:%M:%S'):
        """
        change time
        """
        try:
            date_time = datetime.fromtimestamp(time_stamp)
            return datetime.strftime(date_time, fmt)
        except:
            return ''

    def __call__(self, task_info, table_data_file):
        """
        __call__
        """
        with open(table_data_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # replace to <br/> in html
            content = content.replace("&lt;br/&gt;", "<br/>")
            # add color for failed columns
            red_bg_html = "<td bgcolor='red'>"
            if "failed" in content:
                task_info['status'] = 'Failed'
                content = content.replace("<td>failed</td>",
                                          red_bg_html + "failed</td>")
            content = content.replace('<tr style="text-align: right;">',
                                      '<tr bgcolor="#BEBEBE" style="font-weight:bold;">')

        task_info['table_data'] = content  # insert final table to html

        html = self.generate_html(task_info)
        self.save_html(self.save_path, html)


def main():
    """
    main
    """
    args = parse_args()
    task_info = {}
    task_info['tname'] = args.task_name
    task_info['repo'] = args.repo
    task_info['branch'] = args.branch
    task_info['commit_id'] = args.commit_id
    task_info['build_time'] = time.time()
    task_info['status'] = "none"

    html_render = HtmlContentRender(args.html_output_path)
    html_render(task_info, args.html_table_path)


if __name__ == '__main__':
    main()

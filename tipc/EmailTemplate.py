#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================

"""
@Desc: alarm email 
@File: alarm email 
@Author: guolixin
@Date: 20211207 19:26
"""
import os

MAIL_HEAD_CONTENT = """
<html>
    <body>
        <div style="text-align:center;">
        <a align=center href="BENCHMARK_WEBSITE_ADDR">benchmark前端链接</a><br/><br/>
        <a align=center href="BENCHMARK_PIPELINE_ADDR">任务参考链接</a><br/><br/>
        </div>           
"""

MAIL_TABLE_CONTENT = """
        <HR align=center width="80%" SIZE=1>
        <table style="display:DISPLAY;" border="1" align=center>
        <caption bgcolor="#989898">TABLE_DES</caption>
          <tr bgcolor="#989898" >
TABLE_HEADER_HOLDER
          </tr>
TABLE_INFO_HOLDER
        </table>  
"""

MAIL_TAIL_CONTENT = """
        <HR align=center width="80%" SIZE=1>
        <table border="1" align=center>
        <caption bgcolor="#989898">环境配置</caption>
RUN_ENV_HOLDER
        </table>
    </body>
</html>
"""


class EmailTemplate(object):
    """construct email for benchmark result.
    """
    def __init__(self, env, results, log_path):
        """
        Args:
            env(dict): running environment.
            results(dict):
                {"total": {"header": [table_header0, table_header1, table_header2,]
                           "data": [[{'value':, 'color':, }, {'value':, 'color':, }, {'value':, 'color':, }]
                           ...]}
                "fail": {"header": [table_header0, table_header1, table_header2,]
                        "data": [[{'value':, 'color':, }, {'value':, 'color':, }, {'value':, 'color':, }]
                        ...]}
                ...}
            log_path(str): mail path
        """
        self.env_content = ""
        self.log_path = log_path
        self.results = results
        self.content = ""
        self.__construct_mail_env(env)

    def __construct_mail_env(self, env):
        """
        construct email env content.
        """
        if isinstance(env, dict):
            for k, v in env.items():
                self.env_content += """
                    <tr><td>{}</td><td>{}</td></tr>
                    """.format(k, v)
        return self.env_content

    def __construct_table_info(self, total_info):
        """
        construct table content.
        """
        table_header_info = ""
        table_alarm_info = ""
        if not total_info.get("data"):
            return table_header_info, table_alarm_info
        for header_td in total_info["header"]:
            table_header_info += """
                    <td>{}</td>
                    """.format(header_td)
        for single_info in total_info["data"]:
            if not single_info:
                continue
            table_alarm_info += "\t\t\t<tr>"
            for info_td in single_info:
                table_alarm_info += """
                   <td bgcolor={}>{}</td>
                   """.format(info_td.get('color', 'white'), info_td.get('value'))
            table_alarm_info += "</tr>\n"
        return table_header_info, table_alarm_info

    def __construct_benchmark_pipline_herf(self):
        """
        construct benchmark pipline herf
        """
        CE_SERVER = os.popen("echo ${CE_SERVER}").read()
        BUILD_ID = os.popen("echo${ BUILD_ID}").read()
        BUILD_TYPE_ID = os.popen("echo ${BUILD_TYPE_ID}").read()
        build_link = "%s/viewLog.html?buildId=%s&buildTypeId=%s&tab=buildLog" % (CE_SERVER, BUILD_ID, BUILD_TYPE_ID)
        return build_link
        
    def construct_email_content(self):
        """
        construct email full content.
        """
        content = ""
        build_link = self.__construct_benchmark_pipline_herf()
        website_addr='http://yq01-page-powerbang-table1077.yq01.baidu.com:8988/benchmarks/zongyemian'
        content = MAIL_HEAD_CONTENT.replace('BENCHMARK_PIPELINE_ADDR', build_link).replace(
                'BENCHMARK_WEBSITE_ADDR', website_addr)
        
        disp_dic = {}
        disp_dic["total"] = "任务执行情况汇总表"
        disp_dic["dy_speed"] = "动态图训练吞吐性能变化详情"
        disp_dic["st_speed"] = "静态图训练吞吐性能变化详情"
        disp_dic["dy2st_speed"] = "动转静训练吞吐性能变化详情"
        disp_dic["other"] = "Paddle与竞品天级别汇总表（优:平:劣:paddle失败:竞品失败:无竞品(不差于竞品比例)"
        disp_dic["fail"] =  "失败任务列表"

        for key, value in disp_dic.items():
            if key not in self.results:
                continue
            table_header_info, table_alarm_info = self.__construct_table_info(self.results[key])
            TABLE_TEMP = MAIL_TABLE_CONTENT
            if table_alarm_info:
                TABLE_TEMP =  TABLE_TEMP.replace("TABLE_DES", value).replace(
                        'TABLE_HEADER_HOLDER', table_header_info).replace(
                        'TABLE_INFO_HOLDER', table_alarm_info) 
                if "total" == key:
                    TABLE_TEMP = TABLE_TEMP.replace("<HR align=center width=\"80%\" SIZE=1>", "").replace(
                            "<td>相对标准值下降5%以上个数</td>", "<td bgcolor=\"red\">相对标准值下降5%以上个数</td>").replace(
                            "<td>相对前五次均值下降5%以上个数</td>", "<td bgcolor=\"red\">相对前五次均值下降5%以上个数</td>").replace(
                            "<td>相对稳定版下降5%以上个数</td>", "<td bgcolor=\"red\">相对稳定版下降5%以上个数</td>")
                content += TABLE_TEMP
                  
        content += MAIL_TAIL_CONTENT.replace('RUN_ENV_HOLDER', self.env_content)

        with open(os.path.join(self.log_path, "mail.html"), "w") as f_object:
            f_object.write(content)
        self.content = content

    def construct_weekly_email_content(self):
        """
        construct weekly email full content.
        """
        content = ""
        website_addr='http://yq01-page-powerbang-table1077.yq01.baidu.com:8988/benchmarks/zongyemian'
        content = MAIL_HEAD_CONTENT.replace('BENCHMARK_WEBSITE_ADDR', website_addr).replace(
                "<a align=center href=\"BENCHMARK_PIPELINE_ADDR\">任务参考链接</a><br/><br/>", '')

        disp_dic = {}
        disp_dic["total"] = "周级别执行汇总表"
        disp_dic["dy_speed"] = "动态图训练吞吐性能变化详情"
        disp_dic["st_speed"] = "静态图训练吞吐性能变化详情"
        disp_dic["dy2st_speed"] = "动转静训练吞吐性能变化详情"
        disp_dic["other"] =  "Paddle与竞品天级别汇总表（优:平:劣:paddle失败:竞品失败:无竞品(不差于竞品比例)"

        for key, value in disp_dic.items():
            if key not in self.results:
                continue
            table_header_info, table_alarm_info = self.__construct_table_info(self.results[key])
            TABLE_TEMP = MAIL_TABLE_CONTENT
            if table_alarm_info:
                TABLE_TEMP =  TABLE_TEMP.replace("TABLE_DES", value).replace(
                        'TABLE_HEADER_HOLDER', table_header_info).replace(
                        'TABLE_INFO_HOLDER', table_alarm_info) 
                if "total" == key:
                    TABLE_TEMP = TABLE_TEMP.replace("<HR align=center width=\"80%\" SIZE=1>", "").replace(
                            "<td>相对上周下降5%以上个数</td>", "<td bgcolor=\"red\">相对上周下降5%以上个数</td>")
                content += TABLE_TEMP
                         
        content += MAIL_TAIL_CONTENT.replace('RUN_ENV_HOLDER', self.env_content)

        with open(os.path.join(self.log_path, "mail.html"), "w") as f_object:
            f_object.write(content)
        self.content = content

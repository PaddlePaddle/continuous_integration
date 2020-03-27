#!/bin/bash
set -xe
EMAIL_TITLE="OP CE Pipeline has triggered"
EMAIL_REC="tangtianjie@baidu.com,gaowei22@baidu.com,liuyiqun01@baidu.com"
OP_CE_URL="op ce url:$1/viewLog.html?buildId=$2"
OP_BENCHMARK="op benchmark url:http://yq01-page-powerbang-table1077.yq01.baidu.com:8988/benchmarks/op"
EMAIL_CONTENT="${OP_CE_URL}\n${OP_BENCHMARK}"
echo -e ${EMAIL_CONTENT} | mail -s "$(echo -e  "${EMAIL_TITLE},only for fest\nContent-Type: text/html")" ${EMAIL_REC}
exit 0

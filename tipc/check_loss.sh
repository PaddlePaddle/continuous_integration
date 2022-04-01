#! /bin/bash

push_file=/home/work/bce-client/BosClient.py

logfiles=`find test_tipc -name "*train.log"`
for log in $logfiles; do
   verify_file=$(echo $log | tr '/' '^')
   verify_file=$(echo $verify_file | xargs)
   verify_file=${verify_file}".loss"
   if wget -q https://paddle-qa.bj.bcebos.com/fullchain_ce_loss/$verify_file; then
       grep "\[Train\].*\[Avg\]" $log > losslog
       python check_loss.py $verify_file losslog $log | tee -a loss.result
   else
       grep "\[Train\].*\[Avg\]" $log > $verify_file
       python2 ${push_file} $verify_file paddle-qa/fullchain_ce_loss
   fi
done

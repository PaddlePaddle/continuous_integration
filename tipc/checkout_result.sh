# check_status
set +ex
echo " "

EXIT_CODE=0


zero=0
if [[ -f TIMEOUT ]];then
  timeout_number=$(cat TIMEOUT | wc -l)
  if [ $timeout_number -ne $zero ];then
      #echo "[TIMEOUT] There are $timeout_number models timeout:"
      #cat TIMEOUT
      EXIT_CODE=8
  fi
fi

log_file=RESULT
if [[ ! -f ${log_file} ]];then
  #echo "[ERROR] ${log_file} not exist, all test cases may fail, please check CI task log"
  EXIT_CODE=8
else
  number_lines=$(cat ${log_file} | wc -l)
  failed_line=$(grep -o "Run failed with command" ${log_file}|wc -l)
  if [ $failed_line -ne $zero ]
  then
          echo "[ERROR] There are $number_lines results, but failed number of tests is $failed_line."
          echo "The Following Tests Failed: "
          cat ${log_file} | grep "Run failed with command"
          EXIT_CODE=8
  else
          echo "[SUCCEED] There are $number_lines results, all tipc ${CHAIN} command succeed!"
          
  fi
fi

echo -e "========================================================"
echo " "
echo "Paddle TIPC Tests Finished."
exit ${EXIT_CODE}



# check loss
log_file="loss.result"
if [[ ! -f ${log_file} ]];then
  echo " "
  echo -e "=====================result summary======================"
  echo "${log_file}: No such file or directory"
  echo "[ERROR] ${log_file} not exist, all test cases may fail, please check CI task log"
  echo "========================================================"
  echo " "
  EXIT_CODE=9
else
  number_lines=$(cat ${log_file} | wc -l)
  failed_line=$(grep "[CHECK]" ${log_file} | grep "False" | wc -l)
  zero=0
  if [ $failed_line -ne $zero ]
  then
      echo " "
      echo "Summary Failed Tests ..."
      echo "[ERROR] There are $number_lines cases in ${log_file}, but failed number of tests is $failed_line."
      echo -e "=====================test summary======================"
      echo "The Following Tests Failed: "
      grep "[CHECK]" ${log_file} | grep "False"
      echo -e "========================================================"
      echo " "
      EXIT_CODE=9
  else
      echo "CHECK LOSS SUCCEED!"
  fi
fi

echo "Paddle TIPC Tests Finished."
exit ${EXIT_CODE}

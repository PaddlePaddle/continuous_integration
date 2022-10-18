#1 /bin/bash

source /paddle/python37/bin/active

nohup python3 tool-api.py --callback_addr $1 >> tool-api.log 2>&1 &

for n in `seq 11`; do
	cd tool-$n
		nohup python3 service.py >> tool-$n.log 2>&1 &
	cd -
done

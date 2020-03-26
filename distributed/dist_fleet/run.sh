python dist_fleet_ctr.py --update_method pserver --role pserver --endpoints 127.0.0.1:9121,127.0.0.1:9122 --current_id 0 --trainers 2 > ./ps0.log 2>&1 &
python dist_fleet_ctr.py --update_method pserver --role pserver --endpoints 127.0.0.1:9121,127.0.0.1:9122 --current_id 1 --trainers 2 > ./ps1.log 2>&1 &
python dist_fleet_ctr.py --update_method pserver --role trainer --endpoints 127.0.0.1:9121,127.0.0.1:9122 --current_id 0 --trainers 2 > ./tr0.log 2>&1 &
python dist_fleet_ctr.py --update_method pserver --role trainer --endpoints 127.0.0.1:9121,127.0.0.1:9122 --current_id 1 --trainers 2 > ./tr1.log 2>&1 &

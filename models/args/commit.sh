cp conf/args_batch.conf models/
cp models.py models/
cd models
git rev-parse HEAD >commit_id
git rev-parse HEAD^^^ >commit_id_head
git diff `cat commit_id_head` | grep "diff --git" | awk -F ' b/' '{print $2}'  >change_info

#python commit.py commit_info >change_info
python models.py change_info args_batch.conf >models_info 

cp models_info ../conf/changed_models.conf

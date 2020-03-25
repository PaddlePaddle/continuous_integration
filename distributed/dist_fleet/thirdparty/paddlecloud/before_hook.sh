echo "Run before_hook.sh ..."

# whl package

echo "Getting ImageNet dataset..."
hadoop fs -D fs.default.name=afs://xingtian.afs.baidu.com:9902 -D hadoop.job.ugi=Paddle_Data,Paddle_gpu@2017 -get ./lilong/imagenet_partition/ImageNet.tar.0 ./ &
hadoop fs -D fs.default.name=afs://xingtian.afs.baidu.com:9902 -D hadoop.job.ugi=Paddle_Data,Paddle_gpu@2017 -get ./lilong/imagenet_partition/ImageNet.tar.1 ./ &
hadoop fs -D fs.default.name=afs://xingtian.afs.baidu.com:9902 -D hadoop.job.ugi=Paddle_Data,Paddle_gpu@2017 -get ./lilong/imagenet_partition/ImageNet.tar.2 ./ &
hadoop fs -D fs.default.name=afs://xingtian.afs.baidu.com:9902 -D hadoop.job.ugi=Paddle_Data,Paddle_gpu@2017 -get ./lilong/imagenet_partition/ImageNet.tar.3 ./ &
hadoop fs -D fs.default.name=afs://xingtian.afs.baidu.com:9902 -D hadoop.job.ugi=Paddle_Data,Paddle_gpu@2017 -get ./lilong/imagenet_partition/ImageNet.tar.4 ./ &
hadoop fs -D fs.default.name=afs://xingtian.afs.baidu.com:9902 -D hadoop.job.ugi=Paddle_Data,Paddle_gpu@2017 -get ./lilong/imagenet_partition/ImageNet.tar.5 ./ &
hadoop fs -D fs.default.name=afs://xingtian.afs.baidu.com:9902 -D hadoop.job.ugi=Paddle_Data,Paddle_gpu@2017 -get ./lilong/imagenet_partition/ImageNet.tar.6 ./ &
hadoop fs -D fs.default.name=afs://xingtian.afs.baidu.com:9902 -D hadoop.job.ugi=Paddle_Data,Paddle_gpu@2017 -get ./lilong/imagenet_partition/ImageNet.tar.7 ./ &
hadoop fs -D fs.default.name=afs://xingtian.afs.baidu.com:9902 -D hadoop.job.ugi=Paddle_Data,Paddle_gpu@2017 -get ./lilong/imagenet_partition/ImageNet.tar.8 ./ &
wait
cat ImageNet.tar* > ImageNet.tar
echo "untar..."
tar xvf ImageNet.tar > /dev/null
rm ImageNet.tar*

mkdir models
mkdir utils
mv resnet.py vgg.py models
mv __init__.py models
touch utils/__init__.py
mv reader_cv2.py env.py fp16_utils.py learning_rate.py utility.py img_tool.py utils/

echo "End before_hook.sh ..."



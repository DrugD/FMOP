
# #!/bin/bash
degree=5
for i in `seq 0 9`
do
    # sleep 1 & # 提交到后台的任务
    # echo $i && sleep 1 &
    CUDA_VISIBLE_DEVICES=0 python main.py --internal $((i)) &
done
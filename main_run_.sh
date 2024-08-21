
# #!/bin/bash
degree=5
for i in `seq 15 19`
do
    # sleep 1 & # 提交到后台的任务
    # echo $i && sleep 1 &
    CUDA_VISIBLE_DEVICES=2 python main.py --internal $((i)) &
done
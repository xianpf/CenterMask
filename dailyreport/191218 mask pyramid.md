# 191218 mask pyramid
- anchor free
- box free

# 实验目的
- ## 目的：减少instance mask的数目，且大目标优先
- ## 在4x4或16x16的feature map上逐帧预测 instance mask
- ## 筛选好结果后，加入instance mask的列表
- ## 筛选方法可以是把instance 按缩放大小分配到 32*32、64*64、128*128各个level
- ## 参考借鉴stylegan2的generator
- ## inference的时候按照objectness置信度的顺序来确定优先级

# 挑战
- 如何设计反向传播来保证 25x25 之类的低分辨率级别的优胜劣汰

# 实验准备
- 对target进行精细化处理，把target的各个level的 groud truth 提前处理出来
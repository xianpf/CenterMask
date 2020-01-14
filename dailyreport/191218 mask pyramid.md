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

# mask生长流程
1. 在 size1（7x10）的level上把每个pixel都初始化为一个mask pyramid
2. 为每个mask pyramid 预测其mask （使用mask init conv）
3. 将各个size1（7x10）的mask upsample 到size2（14x20）
4. 找到所有的size2 mask的pixel，这些pixel在任何一个mask pyramid的mask上的值均小于 low threshold = 0.2
5. 在这些找到的pixel上再初始化新的mask pyramid，并为其预测其mask （使用mask init conv）,使用不同target和loss计算方法 但都是L1 loss
6. 在适当的时候，清理这些mask pyrimid that 他们的所有pixel都比不过别的pyramid 进不了前三
7. 循环3 ～ 7



# Ablation studies 几个需要测试的超参数点
- [ ] ```self.mask_init_conv``` 考虑多层  3～5层3*3
- [ ] ```self.score``` 考虑改用mask score rcnn
- [ ] loss 先用L1 loss 后续测试l2 loss 以及cross entropy loss
- [ ] 先不做nms，后面要用nms
- [ ] 在预测各层mask的时候，现在是256+1，考虑256+2、3，添加位置指示点、上层mask等
- [ ] 现在起始点是7x10，考虑4x5，2x4，1x2等等
- [ ] 调节 low threshold
- [ ] 对于mask conv，现在使用的是统一的逐层conv，后面考虑改到统一的按inst pyramid 实际情况使用统一的conv，长远还可探索基于类别的conv，以及conv的加深
- [ ] 针对部分小件target被上层淹没的情况，给予淹没者严厉惩罚
- [ ] mask instance pyramid 部分一定要参考unet的up部分，现在感觉小scale对大一点scale的影响不够有效

# 实验结果
- ## CenterMask的官宣model结果
- ## CenterMask的官宣原配置我run结果
- ## CenterMask的我该参数结果
  - BASE_LR: 0.01
  - IMS_PER_BATCH: 4
  - TEST.IMS_PER_BATCH: 4
  - 结果：
    ```python
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.284
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.444
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.309
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.114
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.305
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.422
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.267
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.417
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.431
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.214
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.474
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.610
    ```



# 问题与解决
- [ ] pyramids 的数量过于庞大，导致cuda memory 溢出
  - 分析：一开始网络不成熟，导致乱生成的情况严重，随着网络成熟应该会好一些
  - 解决办法：分步骤训练。先训练低分辨率的，成熟后导入较高分辨率，逐层训练
  - 解决办法2：设置pyramids数量上限，达到数量后，在backward之前先裁掉mask分值低的那些，从而减少显存占用
- [x] loss 0: 4.67; loss 1: 21.34; loss 2: 80.20 各级的loss差距很大，就导致level 0 的更新反倒不好
  - 分析：可能是各级的pyramid数量不同导致的，把各pyramid的loss求和改为求平均，同时结合上问，控制平衡pyramids数量
  - 解决办法：各pyramid的loss求和改为求平均
  - 解决办法2：限制每级的pyramids数量，筛选掉不行的
- [ ] 一些伪pyramid在自己领地上得分很低，甚至是0分，但是在别人领地上很高，甚至是1.0，这是有问题的。
  - 分析：当时让这些伪pyramid存在，是为了让他们占着地儿，免得大地太空旷，导致下级扩张压力太大，计算量扛不住。这样看来，这些伪pyramid的目标是，填充背景、但不可阻碍下级positive的pyramid的生成。
  - 解决办法：手工设计打分系统，重点惩罚那些在自己领地上表现差的pyramid，让他们及时让路。同时，伪pyramid也有供给抽象object的任务，后期要顾忌抽象能力，调整score打分公式
  - 初版的打分公式 $score = w_r * rootresponse$
- [ ] 由于0 1的格子点问题，导致对应数在各层不是中心对应的，而是左上角对应的，这比较不好，直观上不对应，需要细致分析影响。
- [ ] 同一个image的不同target对应的不同pyramid的输出特征一模一样，没有差异就没有结果
  - 分析：给256层feature施加影响的方式却只有新增一个维度的position指示，这影响太弱了，而且这一层几乎全是0，很多层并不会给它很大的weight
  - 解决办法：生成高斯mask，强行乘到256层的feature上来施加影响
  - 解决办法2:生成高斯mask，并到256层之外，由训练参数来调和其强度
    - 并1层还是没用，考虑全加。可能是因为后续只有1层conv所致，考虑增加conv
- [X] 由于sigmoid的线性区太窄，导致结果非1即0


- [ ] target 的 level化比较失败，考虑修改方式

$$p(x)=\frac{1}{\sqrt{(2\pi)^n det\Sigma}}e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$$

$$p(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{1}{2}(\frac{x-\mu}{\sigma})^2\right)$$
$$p(x)=\exp\left(-ln2(\frac{x-\mu}{\rho})^2\right)$$
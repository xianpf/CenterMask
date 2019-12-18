# 191209 Baseline

# 实验目的
- 跑出几个baseline来作为比较对象
- 部分帮助理解FrosenBatchNorm在干啥
- 
# 实验结果
- ## FCOS 自己报告的最好的结果和提供模型测试结果
  - train_batch_size = 16
  - test_batch_size = 2
  - lr = 0.01
  - 结果：
    ```python
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.387
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.573
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.419
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.226
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.424
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.502
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.325
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.536
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.572
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.367
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.617
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.726
    ```
  - lcation: "run/downloaded FCOS" 
- ## 原原本本的代码跑出来
  - 因为GPU显存不够，没办法跑
  - train_batch_size = 16
  - test_batch_size = 2
  - lr = 0.01
- ## 源代码修改batch size 和 lr
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 1e-4
  - 结果：
    ```python
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.172
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.327
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.161
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.095
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.197
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.226
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.195
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.345
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.374
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.188
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.410
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.519
    ```
  - lcation: "run/fcos_imprv_R_50_FPN_1x/Baseline_lr1en4_191209" 
  ------
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 1e-3
  - 结果：
    ```python
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.292
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.470
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.308
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.163
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.325
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.383
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.270
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.455
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.490
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.283
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.539
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.645
    ```
  - lcation: "run/fcos_imprv_R_50_FPN_1x/Baseline_lr1en3_191210" 
  ------
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 5e-3
  - 结果：
    ```python
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.319
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.492
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.342
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.183
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.358
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.416
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.286
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.484
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.519
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.317
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.572
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.675
    ```
  - lcation: "run/fcos_imprv_R_50_FPN_1x/Baseline_lr5en3_191211" 

- ## 源代码修改batch size 和 lr后from scratch
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 5e-3
  - 结果：
    ```python
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.113
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.193
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.114
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.060
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.121
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.159
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.177
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.305
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.320
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.140
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.329
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.479   
    ```
  - lcation: "run/fcos_imprv_R_50_FPN_1x/Baseline_lr5en3_fbn_scratch_191212" 

- ## 源代码修改batch size 和 lr后from scratch, 不Frozen BN
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 5e-3
  - 结果：
    ```python
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.117
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.199
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.118
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.052
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.123
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.169
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.181
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.313
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.331
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.128
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.344
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.500
    ```
  - lcation: "run/fcos_imprv_R_50_FPN_1x/Baseline_lr5en3_fbn_scratch_191212" 

  - MODEL.BACKBONE.FREEZE_CONV_BODY_AT: -1
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 5e-3
  - 结果：
    ```python
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.139
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.227
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.142
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.072
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.144
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.193
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.198
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.344
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.363
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.166
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.380
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.527
    ```
  - lcation: "run/fcos_imprv_R_50_FPN_1x/Baseline_lr5en3_bn_scratch_nofreeze_191213" 


  - MODEL.BACKBONE.FREEZE_CONV_BODY_AT: -1
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 5e-2
  - 结果：
    ```python
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.001
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.002
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.002
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.003
    ```
  - lcation: "run/fcos_imprv_R_50_FPN_1x/Baseline_lr5en3_bn_scratch_nofreeze_191213" 

- ## 源代码修改batch size 和 lr后from scratch， 并把FrosenBN改为BN
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 0.01

# 最后搞一个☑️的表格

# 最重要还是要回归mask，测试它们的mask表现


# 意识世界的维度可能是无穷多的，我们一般使用有限数量的，预定义的维度来加以模拟
## 一个关键的点是，我们应该只使用最多固定数量8个维度来表示真实世界维度的一个视图view
## 最核心的问题在于
- 新维度的发现机制
- 不同agent的不同维度的统一，合并机制，好互相交流，模拟人类语言、手势协议的过程
- 最终有个library（知识图谱）来统一合并这些视图，从而尽可能地模拟真实世界，

## 一些思维碎片
- 优越感score 长生
- 优越感的各子维度如何产生，随机？
- 一个地主是一个原始维度，多个地主的主成分分析和因子分析造就抽象地主👲
- 正负的调整幅度不一致，正向快，负向慢， 结合丛林法则筛选
- [ ] 自我归因

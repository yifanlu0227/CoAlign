# To train FPV-RCNN & FVoxel-RCNN

To train FPV-RCNN efficiently, it is recommended to first train first stage and then train the whole network.
An example training schedule could be: 

1. Train stage1 for 20 epochs: use fpvrcnn.yaml as reference, make sure that stage2 is inactive

```yaml
model:
  args:
    activate_stage2: False
```

2. Train stage1 and stage2 **for another 20 epochs**: set ```activate_stage2``` to ```True``` and ```epoches``` to ```40```, resume parameters from step 1 and train futher. 


Note that fvoxel-rcnn stage2 seems only accept batchsize to be 1. 

# To Train V2VNet (robust)
V2VNet (robust) is not end-to-end trained. It consists of 3 stages. 

0. create a experiment log folder for V2VNet (robust).

1. Get the pretrain model from V2VNet, name it to ```net_epoch1.pth```.

2. Create a ```config.yaml```, and copy the content from ```pointpillar_v2vnet_robust.yaml```

3. For each stage, rename the lastest (or bestval) model from previous stage to ```net_epoch1.pth```. And change the ```stage, epoches, lr, step_size``` to the corresponding value.  

```yaml
stage: &stage 0
...
  epoches: 15 # stage 0
  # epoches: 20 # stage 1 
  # epoches: 30 # stage 2
...
  lr: 0.001 # stage 0
  # lr: 0.002 # stage 1
  # lr: 0.0002 # stage 2
...
  step_size: [10] # stage 0
  # step_size: [10, 20] # stage 1
  # step_size: [10, 20] # stage 2 
```
-> 
```yaml
stage: &stage 1
...
  # epoches: 15 # stage 0
  epoches: 20 # stage 1 
  # epoches: 30 # stage 2
...
  # lr: 0.001 # stage 0
  lr: 0.002 # stage 1
  # lr: 0.0002 # stage 2
...
  # step_size: [10] # stage 0
  step_size: [10, 20] # stage 1
  # step_size: [10, 20] # stage 2 
```
->
```yaml
stage: &stage 2
...
  # epoches: 15 # stage 0
  # epoches: 20 # stage 1 
  epoches: 30 # stage 2
...
  # lr: 0.001 # stage 0
  # lr: 0.002 # stage 1
  lr: 0.0002 # stage 2
...
  # step_size: [10] # stage 0
  # step_size: [10, 20] # stage 1
  step_size: [10, 20] # stage 2 
```
# To train DiscoNet

1. First train the early model ```pointpillar_early.yaml```

2. Put the bestval model path into ```pointpillar_disconet.yaml```

```yaml
kd_flag:
  teacher_model: point_pillar_disconet_teacher
  teacher_model_config: *model_args
  teacher_path: "Here"
```

3. Use ```train_w_kd.py``` instead of ```train.py```
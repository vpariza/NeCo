### Evaluation

#### Evaluation: Linear and FCN
For linear and FCN fine-tuning we provide a script as well. All configs can be found in `experiments/linear_probing/configs/`. 
For FCN fine-tuning we provide configs under `./pascal/fcn/`.
An exemplary call to evaluate NeCo trained on COCO-Stuff could look like this:
```
python experiments/linear_probing/linear_finetune.py --config_path experiments/linear_probing/configs/coco_stuff/neco.yml
```
We also provide a script to evaluate on the validation dataset at `experiments/linear_probing/eval_linear.py`.

#### Dense NN Retrieval (Hummingbird) Evaluation
To evaluate any of our approaches on the Dense NN Retrieval method you should use the code at `/experiments/dense_nn_ret_eval`
```
python experiments/dense_nn_ret_eval/eval_dense_nn_ret_emb.py  --config_path experiments/dense_nn_ret_eval/pascal/neco.yml --ckpt_path {neco-ckpt-path}
```
for ViTS-16 respectively. 
**Note** that the checkpoint should be the checkpoints saved by Pytorch Lightning during training.

You can define your own configuration files like the one above.

#### Evaluation: Segmentation Maps
You can use one of the models you trained yo generate the segmentation maps over a dataset.
The code responsible for generating the semgentation maps is located in `/experiments/gen_seg_maps`
For example for Pascal VOC you can run the following code.
```
python experiments/gen_seg_maps/gen_seg_maps.py --config_path experiments/gen_seg_maps/configs/pascal/neco.yml
```

#### Evaluation: Overclustering
For overclustering evaluation we provide a script `sup_overcluster.py` under `experiments/overcluster`.
An exemplary call to evaluate a NeCo trained on ImageNet-100 on Pascal VOC could look like this:
```
python experiments/overcluster/sup_overcluster.py --config_path experiments/overcluster/configs/pascal/neco.yml
```
Note that `sup_overcluster.py` can also be used to get fully unsupervised segmentation results by directly clustering to
K=21 classes in the case of Pascal VOC. Configs can be found under `./configs/pascal/k=21/`.

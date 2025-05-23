### Evaluation

#### Evaluation: Linear and FCN
For linear and FCN fine-tuning we provide a script as well. All configs can be found in `experiments/linear_segmentation/configs/`. 
An exemplary call to evaluate NeCo could look like this:
```
python experiments/linear_segmentation/eval_linear.py --config_path experiments/linear_segmentation/configs/coco_stuff/neco.yml
```

#### Evaluation: Overclustering
For overclustering evaluation we provide a script `eval_overcluster.py` under `experiments/overcluster`.
An exemplary call to evaluate NeCo could look like this:
```
python experiments/overcluster/eval_overcluster.py --config_path experiments/overcluster/configs/pascal/neco.yml
```

#### Evaluation: Fully Unsupervised Semantic Segmentation

For fully unsupervised semantic segmentation, use the command below and set the arguments accordingly. For instance:

```
python experiments/overcluster/fully_unsup_seg.py --data_dir "" --ckpt_path ""
```


#### Evaluation: Visual In-Context Learning

Please refer to our repository where we have open-sourced the implementation of the original paper:  
[**"Open Hummingbird Eval"**](https://arxiv.org/abs/2306.01667)  
GitHub: [**vpariza/open-hummingbird-eval**](https://github.com/vpariza/open-hummingbird-eval)

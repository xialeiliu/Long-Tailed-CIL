# Long-Tailed-CIL
<!-- ## News
___ -->
## [ECCV2022]Long-Tailed Class Incremental Learning

This is the official PyTorch implementation of **Long-Tailed Class Incremental Learning**([arxiv](https://arxiv.org/abs/2210.00266)).

## Dataset Prepare

### Cifar100
Download automaticly to `./data`.
### Imagenet
Should be downloaded to `./data`. And the subset list can be generated using the file `./subset.py`.

## Training

Scripts for the main experiments are provided. Their main usage is as follows: 

```
bash ./script_cifar100_no_gridsearch.sh <approach> <gpu> <datasets> <scenario> <cls_num_first_task> <num_tasks> [<results_dir>]
```

where: 
    
* `<approach>` - approach to be used, from the ones in `./src/approaches/`
* `<gpu>` - index of GPU to run the experiment on
* `<datasets>` - dataset to be used (cifar100, imagenet_subset)
* `<scenario>` - specific rehearsal scenario (conv, lt, ltio)
* `<cls_num_first_task>` - the number of classes to be trained in the first base task
* `<num_tasks>` - the number of tasks (include the base task)
* `[<results_dir>]` - results directory (optional), by default it will be `./results`

## Demos


You can train a baseline model of LUCIR by:

```
bash ./script_cifar100_no_gridsearch.sh lucir 0 cifar100 conv 50 11
```

Train a model of LUCIR with 2stage method by:

```
bash ./script_cifar100_no_gridsearch.sh lucir_2stage 0 cifar100 conv 50 11
```

Train a model of LUCIR with 2stage method on shuffled long-tailed scenario by:

```
bash ./script_cifar100_no_gridsearch.sh lucir_2stage 0 cifar100 lt 50 11
```

Train a model of LUCIR with 2stage method on ordered long-tailed scenario by:

```
bash ./script_cifar100_no_gridsearch.sh lucir_2stage 0 cifar100 ltio 50 11
```

Up to now, we implement 2stage methods to three methods (EEIL, LUCIR, PODNET). You can alse use other methods implemented in `./src/approaches`. Scenarios are generated in `./src/datasets/dataloader`

## Reference

If this work is useful for you, please cite us by:
```
@inproceedings{liu2022long,
  title={Long-Tailed Class Incremental Learning},
  author={Liu, Xialei and Hu, Yu-Song and Cao, Xu-Sheng and Bagdanov, Andrew D and Li, Ke and Cheng, Ming-Ming},
  booktitle={European Conference on Computer Vision},
  pages={495--512},
  year={2022},
  organization={Springer}
}
```

## Contact

If you have any questions about this work, please feel free to contact us (xialei AT nankai DOT edu DOT cn or ethanhu AT mail DOT nankai DOT edu DOT cn)

## Thanks

This code is based on [FACIL](https://github.com/mmasana/FACIL)




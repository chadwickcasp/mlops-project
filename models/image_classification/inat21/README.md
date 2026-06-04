---
tags:
- image-classification
- timm
- transformers
library_name: timm
license: cc-by-nc-4.0
---
# Model card for eva02_large_patch14_clip_336.merged2b_ft_inat21

Part of a series of `timm` fine-tune experiments on iNaturalist 2021 competition data (https://github.com/visipedia/inat_comp/tree/master/2021) for higher capacity models. 

Covering 10,000 species, this dataset and these models are fun to explore via the classification widget with pictures from your backyard, but quite a bit smaller than models you can find on iNaturalist website (https://www.inaturalist.org/blog/75633-a-new-computer-vision-model-v2-1-including-1-770-new-taxa).

No extra meta-data was used for training these models (as was the case for the competition), it was a straightfoward fine-tune to explore differences in model pretrain data.

| Model | Top-1 | Top-5 | Img Size (Train) | Paper |
|-------|-------|-------|----------|-------|
| [eva02_large_patch14_clip_336.merged2b_ft_inat21](https://huggingface.co/timm/eva02_large_patch14_clip_336.merged2b_ft_inat21) | 92.05 | 98.01 | 336 | https://arxiv.org/abs/2303.11331 |
| [vit_large_patch14_clip_336.datacompxl_ft_augreg_inat21](https://huggingface.co/timm/vit_large_patch14_clip_336.datacompxl_ft_augreg_inat21) | 91.98 | 98.03 | 336 | https://arxiv.org/abs/2304.14108 |
| [vit_large_patch14_clip_336.laion2b_ft_augreg_inat21](https://huggingface.co/timm/vit_large_patch14_clip_336.laion2b_ft_augreg_inat21) | 91.48 | 97.89 | 336 | https://arxiv.org/abs/2212.07143 |
| [convnext_large_mlp.laion2b_ft_augreg_inat21](https://huggingface.co/timm/convnext_large_mlp.laion2b_ft_augreg_inat21) | 90.95 | 97.68 | 448 (384) | |
| [vit_large_patch14_clip_336.datacompxl_ft_inat21](https://huggingface.co/timm/vit_large_patch14_clip_336.datacompxl_ft_inat21) | 90.85 | 97.68 | 336 | https://arxiv.org/abs/2304.14108 |
| [convnext_large_mlp.laion2b_ft_augreg_inat21](https://huggingface.co/timm/convnext_large_mlp.laion2b_ft_augreg_inat21) | 90.62 | 97.61 | 384 | |
| [vit_large_patch14_clip_336.laion2b_ft_in12k_in1k_inat21](https://huggingface.co/timm/vit_large_patch14_clip_336.laion2b_ft_in12k_in1k_inat21) | 90.29 | 97.44 | 336 | https://arxiv.org/abs/2212.07143 |

## Fine-tune hparams

```
./distributed_train.sh 4 --data-dir /tfds/ --dataset tfds/i_naturalist2021 --amp -j 8 --model vit_large_patch14_clip_224 --img-size 336 --model-kwargs img_size=336  --val-split val --opt adamw --opt-eps 1e-6 --weight-decay .01 --lr 5e-5 -
-warmup-lr 0 --sched-on-updates --clip-grad 1.0 --pretrained -b 48 --num-classes 10000 --grad-accum-steps 8 --layer-decay 0.8
```

```
 ./distributed_train.sh 4 --data-dir /tfds/ --dataset tfds/i_naturalist2021 --amp -j 8 --model eva02_large_patch14_clip_336 --val-split val --opt adamw --opt-eps 1e-6 --weight-decay .01 --lr 5e-5 --warmup-lr 0 --sched-on-updates --clip-gra
d 1.0 --pretrained -b 40 --num-classes 10000 --grad-accum-steps 10 --layer-decay 0.8 --torchcompile
```

## Run Validation 
```
python validate.py /tfds/ --dataset tfds/i_naturalist2021 --model hf-hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21 --split val --amp
```

## Citation

```bibtex
@inproceedings{cherti2023reproducible,
  title={Reproducible scaling laws for contrastive language-image learning},
  author={Cherti, Mehdi and Beaumont, Romain and Wightman, Ross and Wortsman, Mitchell and Ilharco, Gabriel and Gordon, Cade and Schuhmann, Christoph and Schmidt, Ludwig and Jitsev, Jenia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2818--2829},
  year={2023}
}
```

```bibtex
@article{datacomp,
  title={DataComp: In search of the next generation of multimodal datasets},
  author={Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim Entezari, Giannis Daras, Sarah Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga Saukh, Alexander Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont, Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, Ludwig Schmidt},
  journal={arXiv preprint arXiv:2304.14108},
  year={2023}
}
```

```bibtex
@article{EVA02,
  title={EVA-02: A Visual Representation for Neon Genesis},
  author={Fang, Yuxin and Sun, Quan and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2303.11331},
  year={2023}
}
```
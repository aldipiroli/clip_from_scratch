# CLIP from scratch
Implementing from scratch the paper ["Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/abs/2103.00020) ICML 2021 (OpenAI CLIP).

### Clone and install dependencies
``` 
git clone https://github.com/aldipiroli/clip_from_scratch
pip install -r requirements.txt && cd clip
``` 
### Train 
``` 
python train.py config/clip_config.yaml
```

### Inference: zero-shot classification
> Note: Zero shot classification from a model trained on [Flickr8k](http://hockenmaier.cs.illinois.edu/8k-pictures.html) to data from [PascalVOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
``` 
python run_zero_shot_cls.py --config config/clip_config.yaml --ckpt path/to/ckpt
```

### Inference: text-to-image querying
``` 
python run_text_to_img_query.py --config config/clip_config.yaml --ckpt path/to/ckpt --prompt "A picture of a dog ."
```

#### Examples
> Note: Prompt and top-k scores (using cosine similarity between text and image embeddings). Model trained on [Flickr8k](http://hockenmaier.cs.illinois.edu/8k-pictures.html) for 20 epochs. Image queries from the validation set.

![Item 1](images/prompt_1.png)
![Item 1](images/prompt_2.png)
![Item 1](images/prompt_3.png)
![Item 1](images/prompt_4.png)
![Item 1](images/prompt_5.png)
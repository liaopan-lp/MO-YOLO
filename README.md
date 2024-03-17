# MO-YOLO: End-to-End Multiple-Object Tracking Method with YOLO and Decoder
Official implementation of ['MO-YOLO: End-to-End Multiple-Object Tracking Method with YOLO and Decoder'](https://arxiv.org/abs/2310.17170) .
Code will coming soon! 作者被拉去做横向了😵‍💫(The writer has to do some fuxxing things.)
# Introduction
In recent times, the field has witnessed the emergence of end-to-end MOT models based on Transformer architectures, boasting superior performance on datasets such as DanceTracker compared to tracking-by-detection methods. However, the computational demands associated with Transformer models present challenges in terms of both training and deployment, necessitating formidable hardware resources.

In order to address these challenges more effectively, a retrospective analysis of the origins of the Transformer model becomes imperative. The Transformer architecture was initially proposed for the domain of Natural Language Processing (NLP). As researchers continuously refined this model, the GPT emerged as a prominent highlight in the NLP field over the past few years. GPT, exclusively utilizing the decoder component of the Transformer, achieves pre-training and fine-tuning for various language tasks by stacking multiple decoder layers to construct a deep neural network.
![MO_YOLO_00](https://github.com/liaopan-lp/MO-YOLO/assets/69964693/c2b894f3-65ac-4bb5-8a53-402ef47bde42)


Inspired by the success of GPT, this paper adopts a network structure combining the principles of You Only Look Once (YOLO) and RT-DETR, incorporating tracking concepts from MOTR. The resultant model, named MO-YOLO, follows a decoder-based end-to-end MOT approach.

Our code and paper will comming soon.


# Acknowlegment
Thanks for [Ultralytics](https://github.com/ultralytics/ultralytics/) and [MOTR](https://github.com/megvii-research/MOTR).


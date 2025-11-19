# Text-to-Image-Attentional-Generative-Adversarial-Networks

This repository is about assessing an Attentional Generative Adversarial Network (AttnGAN) method which generates images from text. This method can pay attention to the relevant words in the written natural language to synthesize details at
different sub-regions of the image. Due to hardware limitations, this report just evaluates and extends the AttnDCGAN model. To evaluate this method, some text descriptions defined in the original paper are used to compare the generated images by the AttnDCGAN over the CUB dataset. For extension, different kinds of words are tested in the text description to see the behavior of
the system in generating images.

This repo aims to utilize the proposed network in

T. Xu, P. Zhang, Q. Huang, H. Zhang, Z. Gan, X. Huang, and X. He. ”Attngan: Fine-grained text to image generation with attentional
generative adversarial networks.” In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1316-1324. 2018.

to generate high-resolution images from text descriptions. In this model two networks are proposed for this purpose,
AttnGAN and AttnDCGAN. AttnGAN can generate three images with different resolutions, 64x64x3, 128x128x3 and 256x256x3. However, the AttnDCGAN can just generate the
64x64x3 image. Here, AttnGAN is built over the CUB and COCO dataset and AttnDCGAN is just built over the CUB dataset.

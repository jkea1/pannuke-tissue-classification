### Models
#### 1. ResNet-50
Why we chose ResNet-50 for this project:
- It’s a very common baseline backbone in image classification and medical imaging.
- It’s much more capable than ResNet-18/34, but still much lighter than ResNet-101/152.
- It has a good balance of accuracy, training speed, and GPU memory use.
- It makes a fair comparison with DenseNet-121 and Swin-T without making the experiment unnecessarily heavy.
- There are readily available ImageNet-pretrained weights, so we can fine-tune instead of training from scratch.

### References
[panNuke dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/)  
[pathml](https://github.com/Dana-Farber-AIOS/pathml)

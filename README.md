## Setup
git clone https://github.com/gulvarol/smplpytorch.git
cd smplpytorch
`conda env update -f environment.yml` in an existing environment
    
### 2. Download SMPL pickle files (Already in the repo so no need to download)
  * Download the models from the [SMPL website](http://smpl.is.tue.mpg.de/) by choosing "SMPL for Python users". Note that you need to comply with the [SMPL model license](http://smpl.is.tue.mpg.de/license_model).
  * Extract and copy the `models` folder into the `smplpytorch/native/` folder.

## Acknowledgements
The code **largely** builds on the [manopth](https://github.com/hassony2/manopth) repository from [Yana Hasson](https://github.com/hassony2), which implements the [MANO](http://mano.is.tue.mpg.de) hand model [\[2\]](#references) layer.

The code is a PyTorch port of the original [SMPL](http://smpl.is.tue.mpg.de) model from [chumpy](https://github.com/mattloper/chumpy). It builds on the work of [Loper](https://github.com/mattloper) et al. [\[1\]](#references).

The code [reuses](https://github.com/gulvarol/smpl/pytorch/rodrigues_layer.py) [part of the code](https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py) by [Zhang Xiong](https://github.com/MandyMo) to compute the rotation utilities.

If you find this code useful for your research, please cite the original [SMPL](http://smpl.is.tue.mpg.de) publication:

```
@article{SMPL:2015,
    author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
    title = {{SMPL}: A Skinned Multi-Person Linear Model},
    journal = {ACM Trans. Graphics (Proc. SIGGRAPH Asia)},
    number = {6},
    pages = {248:1--248:16},
    volume = {34},
    year = {2015}
}
```

## References

\[1\] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black, "SMPL: A Skinned Multi-Person Linear Model," SIGGRAPH Asia, 2015.

\[2\] Javier Romero, Dimitrios Tzionas, and Michael J. Black, "Embodied Hands: Modeling and Capturing Hands and Bodies Together," SIGGRAPH Asia, 2017.

## Setup
git clone https://github.com/gulvarol/smplpytorch.git
cd smplpytorch
`conda env update -f environment.yml` in an existing environment
    
### 2. Download SMPL pickle files (Already in the repo so no need to download)
  * Download the models from the [SMPL website](http://smpl.is.tue.mpg.de/) by choosing "SMPL for Python users". Note that you need to comply with the [SMPL model license](http://smpl.is.tue.mpg.de/license_model).
  * Extract and copy the `models` folder into the `smplpytorch/native/` folder.
  * Have both male, female and neutral there.

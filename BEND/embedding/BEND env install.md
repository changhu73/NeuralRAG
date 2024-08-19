```bash
conda create -n bend python=3.10

# install pytorch=2.2.1 according to https://pytorch.org/get-started/previous-versions/
# I have cuda 11.8, so
# conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# remember to test the cuda via "torch.cuda.is_available()"

# install packages
pip install -r requirements.txt 

# install BEND
git clone https://github.com/frederikkemarin/BEND.git
cd BEND
pip install -e .

# remove triton to avoid error
pip uninstall triton
```




# install env
conda create -n "recommender" python=3.8 ipython

# add a new python env to notebook launcher
conda install jupyter ipykernel
python -m ipykernel install --user --name=recommender --display-name="Python3.8 (recommender)"

# install packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

pip3 freeze > requirements.txt

# update python env path in the main.pbs
echo $CONDA_PREFIX

# check and update batch size to fit memory
!nvidia-smi

# make sure the version !pip install gradio==3.43.1
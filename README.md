
# How to start
git clone this repository
```
git clone https://github.com/Vulpes94/quoridor-zero.git
```

```sh
conda create -n ai-rl -y python=3.11
conda activate ai-rl
pip install -r requirements.txt
jupytext --to ipynb */*.py 
jupyter notebook
```
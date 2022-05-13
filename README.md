# GCRL-min(AoI)
Additional materials for paper "AoI-minimal UAV Crowdsensing by Model-based
Graph Convolutional Reinforcement Learning" accepted to INFOCOM 2022.

## :page_facing_up: Description
Mobile Crowdsensing (MCS) with smart devices has become an appealing paradigm for urban sensing.With the development of 5G-and-beyond technologies, unmanned aerial vehicles (UAVs) become possible for real-time applications, including wireless coverage, search and even disaster response. In this paper, we consider to use a group of UAVs as aerial base stations (BSs) to move around and collect data from multiple MCS users, forming a UAV crowdsensing campaign (UCS). Our goal is to maximize the collected data, geographical coverage whiling minimizing the age-of-information (AoI) of all mobile users simultaneously, with efficient use of constrained energy reserve. We propose a model-based deep reinforcement learning (DRL) framework called ”GCRL-min(AoI)”, which mainly consists of a novel model-based Monte Carlo tree search (MCTS) structure based on state-of-the- art approach MCTS (AlphaZero). We further improve it by adding a spatial UAV-user correlation extraction mechanism by a relational graph convolutional network (RGCN), and a next state prediction module to reduce the dependance of experience data. Extensive results and trajectory visualization on three real human mobility datasets in Purdue University, KAIST and NCSU show that GCRL-min(AoI) consistently outperforms five
baselines, when varying different number of UAVs and maximum coupling loss in terms of four metrics.

## :wrench: Installation
1. Clone repo
    ```bash
    git clone https://github.com/BIT-MCS/GCRL-min-AoI.git
    cd GCRL-min-AoI
    ```
2. Install dependent packages
    ```sh
    # system-env
    sudo apt-get install libgeos++-dev libproj-dev
    
    # python-env
    conda create -n mcs python==3.8
    conda activate mcs
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 tensorboard future
    conda install --channel conda-forge cartopy
    pip install -r requirements.txt
    
    # Install movingpandas
    mkdir requirements && cd requirements
    git clone https://github.com/anitagraser/movingpandas.git
    python setup.py develop
    ```


## :computer: Training

Train our solution
```bash
python train_our_policy.py --overwrite --output_dir logs/debug
```


Train our solution with trajectory visualization for debugs

```sh
python train_our_policy.py --overwrite --test_after_every_eval --vis_html --plot_loop --moving_line --output_dir logs/debug
```

## :checkered_flag: Testing

Test with the trained models 

```
python test_our_policy.py --vis_html --plot_loop --moving_line --model_dir logs/debug
```

Random test the env

```
python test_random.py --overwrite --vis_html --plot_loop --moving_line --output_dir logs/debug
```

## :clap: Reference
- https://github.com/vita-epfl/CrowdNav
- https://github.com/ChanganVR/RelationalGraphLearning


## :scroll: Acknowledgement

This paper was sponsored in part by National Natural Science Foundation of China (No. U21A20519 and 62022017), and in part by the National Research and Development Program of China under Grant 2019YQ1700. Corresponding author: C. H. Liu.
<br>
Corresponding author: Chi Harold Liu.

## :e-mail: Contact

If you have any question, please email `3120215520@bit.edu.cn`.

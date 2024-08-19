# Video Anomaly Detection with Probabilistic Graphical Models

**Note**: This repository contains randomly generated ground truth annotations of objects that serve as toy example to showcase the use of our code. This includes bounding box coordinates (in YOLO format) and track IDs. By default, the same annotations are used for training and testing which implies that all objects will get a very high probability score (1.0) assigned during inference. This shall enable an easy verification as to whether the code has run correctly.


## Welcome!
This repository serves as accompanying material to our paper **'Bounding Boxes and Probabilistic Graphical Models: Video Anomaly Detection Simplified'** which has been accepted for publication at [GCPR 2024](https://www.gcpr-vmv.de/year/2024) and invited for a poster presentation:

```bibtex
@misc{Siemon-GCPR-2024,
      title={Bounding Boxes and Probabilistic Graphical Models: Video Anomaly Detection Simplified}, 
      author={Mia Siemon and Thomas B. Moeslund and Barry Norton and Kamal Nasrollahi},
      year={2024},
      eprint={2407.06000},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.06000}, 
}
```

Here, we share the code used to produce the announced results using the spatio-temporal model version of our discrete Bayesian Network. In addition to this, we also share the graphical structure of the spatial counterpart. Both network versions are defined in ```./core/network.py```.

The code can be executed out of the box by calling the following command:
```bash
./run_script.sh
```

This script will then call ```parse_script.py``` with a couple of arguments which can be specified by the user (calling ```python parse_script.py -h``` will display a short description of all those arguments).

**Note**: Upon successful execution, a text file will be populated with the results and saved in ```./output/results/``` . The formatting of that file will be explained further below.

Overall, the project is composed of the following components:
```
- data/var_lib.py : Contains declarations of the project's global variables
- main_script.py : The main script of the project
- core/core.py : Scripts with core functions responsible for the main calculations
- core/discrete.py : Helper functions creating the statistical foundation for the main calculations
- data/dataloader.py : Location of the function responsible for loading the meta-data from the preprocessed frames (bounding box locations, classes, track IDs, confidence scores)
```

## How to Run with Own Data
This is a toy example in which the same set of ground truth annotations is used for training and testing the model. This implies that all objects will be labelled with probability 1.0 (very likely / normal) and none as anomalous. Because of this, current placeholder values (constant) in the file ```./data/dataloader.py``` have to be replaced with actual meta-data by the user.

Further, the following adjustments should be made:
```
- data/var_lib.py : Adjust the directories according to the desired structure
- parse_script.py : Specify the directory in which the results are to be stored (currently: ./output/results/)
- dataloader.py :  Adjust the functions for calculating the average confidence threshold / loading the meta-data (**Note**: The code in this repository is solely based on lists of images and their corresponding meta-data, i.e., *no images are needed*.)
```

If you want to experiment with your own custom network constellations, approriate changes need to be made to the scripts in ```./core/``` .


Please note that we do not publish any evaluation script in this repository as we kindly refer to the code published by Georgescu et al. at https://github.com/lilygeorgescu/AED/tree/master/evaluation alongside their work:

```bibtex
@article{Georgescu-TPAMI-2021, 
  author={Georgescu, Mariana Iuliana and Ionescu, Radu and Khan, Fahad Shahbaz and Popescu, Marius and Shah, Mubarak}, 
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},  
  title={A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3074805}}
```


### Results:

Default path: ```(./output/results/results--object-scores--[model_name].txt)```

Format: (filename, [ N, [scores], [classes], [boxes], [conf_scores], [track_ids] ])
```
*N*:  number of objects in the frame
*scores*: list of object (anomaly) scores (**Note**: All detected objects are scored with a probability value indicating how likely this object is to appear in the frame - including normal *and* abnormal objects)
*classes* : list of object classes
*boxes* : list of corresponding bounding boxes of format [x_min, y_min, x_max, y_max]
*conf_scores* : object detection confidence scores
*track_ids* : unique track IDs of the objects
```

**Note**: The order of all lists (scores, classes, etc.) is fixed, i.e., the first object in a frame can be identified via the list values at index 0, the second object via index 1, etc.

For example: ('02.jpg', [1, [1.0], ['person'], [[82, 155, 122, 245]], [1.0], [0]])

### Explainability of Results
In ```./core/core.py``` we provide a function to explain the obtained results in terms of the modelled variables in the graph, called ```explain_anomaly_score(...)```. It will be automatically called when the value 1 is passed as global argument to ```parse_script.py``` in ```./run_project.sh``` printing all relevant details to the terminal. By default, this value is set to 0. Example visualizations are given in the main paper and its supplementary material.

## Run on docker

The following commands are designed to be executed from the root of the git repository

Build docker image

```bash
docker build -t vad-with-pgms docker/
```

Enter docker in interactive mode
```bash
docker run --mount type=bind,source=$PWD,destination=/workdir -it vad-with-pgms /bin/bash
```

Execute the project

```bash
./run_project.sh
```

## Citing VAD with PGMs
If you use the code of this repo in your research or wish to refer to the results we published in our paper, please use the following BibTeX entry to cite it:

```bibtex
@misc{Siemon-GCPR-2024,
      title={Bounding Boxes and Probabilistic Graphical Models: Video Anomaly Detection Simplified}, 
      author={Mia Siemon and Thomas B. Moeslund and Barry Norton and Kamal Nasrollahi},
      year={2024},
      eprint={2407.06000},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.06000}, 
}
```


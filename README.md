# Skeleton-based Human Action Recognition

<div align="left">
  <a href="https://creativecommons.org/licenses/by-nc/4.0/">
    <img src="https://img.shields.io/badge/License-CC.BY.NC.4.0-blue.svg" height="20">
  </a>
	
  <a href="https://ieeexplore.ieee.org/abstract/document/9412091">
    <img src="http://img.shields.io/badge/paper-ICPR.2020-B31B1B.svg" height="20" >
  </a>
  <a href="https://ieeexplore.ieee.org/abstract/document/9413860">
    <img src="http://img.shields.io/badge/paper-ICASSP.2021-B31B1B.svg" height="20" >
  </a>
  <a href="https://ieeexplore.ieee.org/abstract/document/9534440">
    <img src="http://img.shields.io/badge/paper-IJCNN.2021-B31B1B.svg" height="20" >
  </a>
  <a href="https://arxiv.org/abs/2106.04332">
    <img src="http://img.shields.io/badge/paper-arxiv.2106.04332-B31B1B.svg" height="20" >
  </a>
	
	
  <sup>
</div>
	
This repository provides the implementation of the baseline method ST-GCN [[1]](#1), its extension 2s-AGCN [[2]](#3),
and our proposed methods TA-GCN [[3]](#3), PST-GCN[[4]](#4), ST-BLN [[5]](#5), and PST-BLN [[6]](#6) for skeleton-based human action recognition. 
Our proposed methods are built on top of ST-GCN to make it more efficient in terms of number of model parameters and floating point operations. 

The application of ST-BLN and PST-BLN methods are also evaluated on facial expression recognition methods for 
landmark-based facial expression recognition task in our paper [[6]](#6), and the implementation can be found in [FER_PSTBLN_MCD](https://github.com/negarhdr/FER_PSTBLN_MCD).


This implementation is modified based on the [OpenMMLAB toolbox](
https://github.com/open-mmlab/mmskeleton/tree/b4c076baa9e02e69b5876c49fa7c509866d902c7), and the [2s-AGCN](https://github.com/lshiwjx/2s-AGCN) repositories.

This project is funded by the [OpenDR European project]((https://opendr.eu/)) and the implementations are also integrated in OpenDR toolkit which will be publicly available soon. 
# Data Preparation

 - Download the raw data from [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) and [Skeleton-Kinetics](https://github.com/yysijie/st-gcn). Then put them under the data directory:
 
        -data\  
          -kinetics_raw\  
            -kinetics_train\
              ...
            -kinetics_val\
              ...
            -kinetics_train_label.json
            -keintics_val_label.json
          -nturgbd_raw\  
            -nturgb+d_skeletons\
              ...
            -samples_with_missing_skeletons.txt
            

[https://github.com/shahroudy/NTURGB-D]: NTU-RGB+D
[https://github.com/yysijie/st-gcn]: Skeleton-Kinetics

 - Preprocess the data with
  
    `python data_gen/ntu_gendata.py`
    
    `python data_gen/kinetics-gendata.py.`

 - Generate the bone data with: 
    
    `python data_gen/gen_bone_data.py`
     
# Training & Testing

Modify config files based on your experimental setup and run the following scripts:


    `python main.py --config ./config/nturgbd-cross-view/stgcn/train_joint_stgcn.yaml`

    `python main.py --config ./config/nturgbd-cross-view/stgcn/train_bone_stgcn.yaml`
To ensemble the results of joints and bones, first run test to generate the scores of the softmax layer. 

    `python main.py --config ./config/nturgbd-cross-view/stgcn/test_joint_stgcn.yaml`

    `python main.py --config ./config/nturgbd-cross-view/stgcn/test_bone_stgcn.yaml`

Then combine the generated scores with: 

    `python ensemble.py` --datasets ntu/xview
 
 The shell scripts for training and testing each of the methods are also provided. For example, for training the ST-GCN method you need to run:
    
    `sh run_stgcn.sh`
 
	
# Demo
All the aforementioned methods are also integerated in the [OpenDR toolkit](https://github.com/opendr-eu/opendr) along with a webcam [demo code](https://github.com/opendr-eu/opendr/tree/master/projects/perception/skeleton_based_action_recognition/demos). 
In this demo, we use [light-weight OpenPose](https://github.com/opendr-eu/opendr/tree/master/src/opendr/perception/pose_estimation/lightweight_open_pose) [[10]](#10), which is integerated in the toolkit as well, to extract skeletons from each input frame and then we feed a sequence of 300 skeletons to a pre-trained ST-GCN-based model in this toolkit. 

https://user-images.githubusercontent.com/36771997/163422313-262427ab-57e7-476e-92a4-142b074a2d07.mp4


	
# Citation
Please cite the following papers if you use any of the proposed methods implemented in this repository in your reseach.
##### TA-GCN
	@inproceedings{heidari2021tagcn,
          title={Temporal attention-augmented graph convolutional network for efficient skeleton-based human action recognition},
          author={Heidari, Negar and Iosifidis, Alexandros},
          booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
          pages={7907--7914},
          year={2021},
          organization={IEEE}
    }
##### PST-GCN
    @inproceedings{heidari2021pstgcn,
          title={Progressive Spatio-Temporal Graph Convolutional Network for Skeleton-Based Human Action Recognition},
          author={Heidari, Negar and Iosifidis, Alexandras},
          booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
          pages={3220--3224},
          year={2021},
          organization={IEEE}
        }
##### ST-BLN
    @inproceedings{heidari2021stbln,
          title={On the spatial attention in spatio-temporal graph convolutional networks for skeleton-based human action recognition},
          author={Heidari, Negar and Iosifidis, Alexandros},
          booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
          pages={1--7},
          year={2021},
          organization={IEEE}
        }
##### PST-BLN
    @article{heidari2021pstbln,
          title={Progressive Spatio-Temporal Bilinear Network with Monte Carlo Dropout for Landmark-based Facial Expression Recognition with Uncertainty 	   Estimation},
          author={Heidari, Negar and Iosifidis, Alexandros},
          journal={arXiv preprint arXiv:2106.04332},
          year={2021}
        }

	
	
# Acknowledgement

This work was supported by the European Union’s Horizon
2020 Research and Innovation Action Program under Grant
871449 (OpenDR).
<div align="left">
<img src="https://user-images.githubusercontent.com/16520105/123549590-6a9f4b00-d772-11eb-998a-ed4c70133617.png" height="70"> <img src="https://user-images.githubusercontent.com/16520105/123549536-31ff7180-d772-11eb-9c81-6cc98b7d2e1e.png" height="70">
</div>
  

# Contact
For any questions, feel free to contact: `negar.heidari@ece.au.dk`


## References

<a id="1">[1]</a> 
[Yan, S., Xiong, Y., & Lin, D. (2018, April). Spatial temporal graph convolutional networks for skeleton-based action 
recognition. In Proceedings of the AAAI conference on artificial intelligence (Vol. 32, No. 1).](
https://arxiv.org/abs/1609.02907)

<a id="2">[2]</a> 
[Shi, Lei, et al. Two-stream adaptive graph convolutional networks for skeleton-based action recognition. Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition. 2019.](
https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html)


<a id="3">[3]</a> 
[Heidari, Negar, and Alexandros Iosifidis. "Temporal attention-augmented graph convolutional network for efficient skeleton-based human action recognition." 2020 25th International Conference on Pattern Recognition (ICPR). IEEE, 2021.](https://ieeexplore.ieee.org/abstract/document/9412091)


<a id="4">[4]</a> 
[Heidari, Negar, and Alexandras Iosifidis. "Progressive Spatio-Temporal Graph Convolutional Network for Skeleton-Based Human Action Recognition." ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021.](https://ieeexplore.ieee.org/abstract/document/9413860)

<a id="5">[5]</a> 
[Heidari, N., & Iosifidis, A. (2020). On the spatial attention in Spatio-Temporal Graph Convolutional Networks for 
skeleton-based human action recognition. arXiv preprint arXiv: 2011.03833.](https://arxiv.org/abs/2011.03833)

<a id="6">[6]</a> 
[Heidari, Negar, and Alexandros Iosifidis. "Progressive Spatio-Temporal Bilinear Network with Monte Carlo Dropout for Landmark-based Facial Expression Recognition with Uncertainty Estimation." arXiv preprint arXiv:2106.04332 (2021).](https://arxiv.org/abs/2106.04332)

<a id="7">[7]</a> 
[Shahroudy, A., Liu, J., Ng, T. T., & Wang, G. (2016). Ntu rgb+ d: A large scale dataset for 3d human activity analysis.
 In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1010-1019).](
 https://openaccess.thecvf.com/content_cvpr_2016/html/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.html)

<a id="8">[8]</a>
[Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier, C., Vijayanarasimhan, S., ... & Zisserman, A. (2017). 
The kinetics human action video dataset. arXiv preprint arXiv:1705.06950.](https://arxiv.org/pdf/1705.06950.pdf) 

<a id="9">[9]</a>
[Cao, Z., Simon, T., Wei, S. E., & Sheikh, Y. (2017). Realtime multi-person 2d pose estimation using part affinity 
fields. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7291-7299).](
https://openaccess.thecvf.com/content_cvpr_2017/html/Cao_Realtime_Multi-Person_2D_CVPR_2017_paper.html)
	
<a id="10">[10]</a>
[Osokin, Daniil (2017). Real-time 2d multi-person pose estimation on cpu: Lightweight openpose. arXiv preprint arXiv:1811.12004.](
https://arxiv.org/abs/1811.12004)
	


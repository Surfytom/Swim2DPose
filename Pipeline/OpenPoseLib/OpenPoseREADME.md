# OpenPose

The [OpenPose repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose) has represented the first real-time multi-person system to jointly detect human body, hand, facial, and foot keypoints (in total 135 keypoints) on single images. 
It is authored by Ginés Hidalgo, Zhe Cao, Tomas Simon, Shih-En Wei, Yaadhav Raaj, Hanbyul Joo, and Yaser Sheikh. It is maintained by Ginés Hidalgo and Yaadhav Raaj. OpenPose would not be possible without the CMU Panoptic Studio dataset. We would also like to thank all the people who have helped OpenPose in any way.

# OpenPose installation

Follow step by step installation guide within [Dockerfile](https://github.com/Surfytom/Swim2DPose/blob/main/PipeLine/OpenPoseLib/OpenPoseREADME.md)

## Using OpenPose

To use OpenPose Run the following command within the pipeline container command line:

```
python3 Pipeline/main.py -m OpenPose
```

If you don't have the pipeline container installed please follow the setup guide in the main [readme](https://github.com/Surfytom/Swim2DPose/blob/main/README.md#pipeline).


## Citation
Please cite these papers in your publications if OpenPose helps your research. All of OpenPose is based on [OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1812.08008), while the hand and face detectors also use [Hand Keypoint Detection in Single Images using Multiview Bootstrapping](https://arxiv.org/abs/1704.07809) (the face detector was trained using the same procedure as the hand detector).

    @article{8765346,
      author = {Z. {Cao} and G. {Hidalgo Martinez} and T. {Simon} and S. {Wei} and Y. A. {Sheikh}},
      journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
      title = {OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      year = {2019}
    }

    @inproceedings{simon2017hand,
      author = {Tomas Simon and Hanbyul Joo and Iain Matthews and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Hand Keypoint Detection in Single Images using Multiview Bootstrapping},
      year = {2017}
    }

    @inproceedings{cao2017realtime,
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      year = {2017}
    }

    @inproceedings{wei2016cpm,
      author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Convolutional pose machines},
      year = {2016}
    }

Paper links:
- OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields:
    - [IEEE TPAMI](https://ieeexplore.ieee.org/document/8765346)
    - [ArXiv](https://arxiv.org/abs/1812.08008)
- [Hand Keypoint Detection in Single Images using Multiview Bootstrapping](https://arxiv.org/abs/1704.07809)
- [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1611.08050)
- [Convolutional Pose Machines](https://arxiv.org/abs/1602.00134)



## License
OpenPose is freely available for free non-commercial use, and may be redistributed under these conditions. Please, see the [license](./LICENSE) for further details. Interested in a commercial license? Check this [FlintBox link](https://cmu.flintbox.com/#technologies/b820c21d-8443-4aa2-a49f-8919d93a8740). For commercial queries, use the `Contact` section from the [FlintBox link](https://cmu.flintbox.com/#technologies/b820c21d-8443-4aa2-a49f-8919d93a8740) and also send a copy of that message to [Yaser Sheikh](mailto:yaser@cs.cmu.edu).

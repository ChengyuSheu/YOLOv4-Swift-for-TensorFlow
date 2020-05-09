# YOLOv4-TensorFlow-swift-apis
![](https://img.shields.io/static/v1?label=Swift&message=5.0&color=red)
![](https://img.shields.io/static/v1?label=TensorFlow&message=0.9&color=yellow)

This is a minimum implementation of YOLOv4 in Swift for TensorFlow, S4TF. Everything, including helper functions, is implemented in Swift. This repository is a complete example for researchers/engineers to get a first insight of pure S4TF, even though it supports Python libraries =).

Requirement:
- Swift for TensorFlow, 0.9. (used a lot of hacked tricks)
- swift-jupyter + SPM

Achieved:
- EXACT OUTPUT like Darknet
- faster than Darknet on CPU

feature:
- [x] CORE
  - [x] yolov4 structure
  - [ ] post-processing, YOLO layer, NMS
- [ ] I/O
  - [x] Darknet I/O
  - [x] Swift load Images
  - [ ] Swift Visualization
- [ ] Train
- [ ] refactor code (it's my first swift project, hacky things need to be improved.

NN components:
- [x] SPP, Spatial Pyramid Pooling
- [x] CSP, Cross-Stage-Partial-connections
- [x] mish, activation function
- [x] Darknet-padding methods(used in Convolutional_layer)

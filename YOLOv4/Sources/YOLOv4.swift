import TensorFlow
import Foundation

public struct YOLO<T: TensorFlowFloatingPoint>: ParameterlessLayer {
    @noDerivative public var anchors: Tensor<Float>
    @noDerivative public var imSize: [Int]
    @noDerivative public var cls: Int
    @noDerivative public var anc: Int
    @noDerivative public var scale: Float
    @noDerivative public var test: Bool
    public init(imageSize: [Int],
                anchors ans: Tensor<Float>,
                cls c: Int = 80,
                scale s: Float = 1
               ) {
        imSize = imageSize
        anchors = ans
        anc = anchors.shape[0]
        cls = c
        scale = s
        test = false
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let b = input.shape[0]
        let h = input.shape[1]
        let w = input.shape[2]
        let c = input.shape[3]
                
        precondition(anc*(cls+5) == c, "channel number incorrect.")
        //var stride = Tensor<Float>([Float(imSize[0])/Float(h), Float(imSize[1])/Float(w)])
        //var stride = Tensor<Float>([1/Float(h), 1/Float(w)])
        var inputView: Tensor<Float> = input.reshaped(to: [b, h, w, anc, cls + 5])
        var s: [Int32] = [2,2,1,Int32(cls)]
        var features = inputView.split(sizes: Tensor<Int32>(s), alongAxis: -1)
        var box_centers = sigmoid(features[0]) * scale - (scale - 1) / 2
        var confidence = sigmoid(features[2])
        var classes = sigmoid(features[3])
        var box_sizes = features[1]
    
       

        if (!test) {
            
            var grid_x  = Tensor<Float>(rangeFrom: 0, to: Float(w), stride: 1)
            var grid_y  = Tensor<Float>(rangeFrom: 0, to: Float(h), stride: 1)
            var x_offset = grid_y.reshaped(to: [1,1,w,1,1]).tiled(multiples: [1,h,1,anc,1])
            var y_offset = grid_x.reshaped(to: [1,h,1,1,1]).tiled(multiples: [1,1,w,anc,1])
            var offset = x_offset.concatenated(with: y_offset, alongAxis: -1)

            box_centers = (box_centers + offset) / Tensor<Float>([Float(h), Float(w)])
            box_sizes =  exp(box_sizes) * anchors

        }     
        
        return box_centers.concatenated(with: box_sizes, alongAxis: -1)
            .concatenated(with: confidence, alongAxis: -1)
            .concatenated(with: classes, alongAxis: -1)
            .reshaped(to: [1,h,w,c])
    }
}

public struct CSPDarknet53<T: TensorFlowFloatingPoint>: Layer {
  public var base: ConvBN<T>
  public var cspBlock: [CSPBlock<T>]

  public init() {
    base = ConvBN<T>(inFilters: 3, outFilters: 32, kernelSize: 3)
    cspBlock = [CSPBlock<T>(inFilters: 32, outFilters: 64, repeatDense: 1, expand: true),
                           CSPBlock<T>(inFilters: 64, outFilters: 128, repeatDense: 2),
                           CSPBlock<T>(inFilters: 128, outFilters: 256, repeatDense: 8), //output_level = 54
                           CSPBlock<T>(inFilters: 256, outFilters: 512, repeatDense: 8), 
                           CSPBlock<T>(inFilters: 512, outFilters: 1024, repeatDense: 4)] //output_level = 85
  }
  
  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    return input |> base |> cspBlock
  }
}


public struct CSPDarknet53ImageNet<Scalar: TensorFlowFloatingPoint>: Layer {
  public var base = CSPDarknet53<Float>()
  public var conv = Conv2D<Float>(filterShape: (1, 1, 1024, 1000))
  public var avgpool = GlobalAvgPool2D<Float>()
      public init() {
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
      return softmax(
                  (input |> base |> avgpool).reshaped(to: [input.shape[0], 1, 1, 1024])
                  |> conv)
  }
}

public struct YOLOv4<T: TensorFlowFloatingPoint>: Layer {
  public var backbone: CSPDarknet53<Float>
  public var spp: SPP<Float>
  @noDerivative public var outFilters: Int
  public var convBNs1: [ConvBN<Float>]
  public var convBNs2: [ConvBN<Float>] 
  public var up: UpSampling2D<Float>
  public var convBN3: ConvBN<Float>
  public var convBN2: ConvBN<Float>
  public var conv_ml1: ConvBN<Float>
  public var conv_ml2: ConvBN<Float>
  public var conv_ml3: ConvBN<Float>
  public var conv_ml4: ConvBN<Float>
  public var oconv1: ConvBN<Float>
  public var oconv2: ConvBN<Float>
  public var oconv3: ConvBN<Float>
  public var yolo1: Conv2D<Float>
  public var yolo2: Conv2D<Float>
  public var yolo3: Conv2D<Float>  
  public var convBNssss0: [ConvBN<Float>]
  public var convBNssss1: [ConvBN<Float>]
  public var convBNssss2: [ConvBN<Float>]
  public var convBNssss3: [ConvBN<Float>]
   
    public var yoloHead1: YOLO<Float>
    public var yoloHead2: YOLO<Float>
    public var yoloHead3: YOLO<Float>
    
    
  public init(cls: Int){
    //activation=leaky
    outFilters = (cls + 1 + 4) * 3
    backbone = CSPDarknet53<Float>()
    spp = SPP<Float>()
    convBNs1 = [ConvBN<Float>(inFilters: 1024, outFilters: 512, kernelSize: 1, activation: myLeakyRelu),
              ConvBN<Float>(inFilters: 512, outFilters: 1024, kernelSize: 3, activation: myLeakyRelu),
              ConvBN<Float>(inFilters: 1024, outFilters: 512, kernelSize: 1, activation: myLeakyRelu)]
      
   convBNs2 = [ConvBN<Float>(inFilters: 2048, outFilters: 512, kernelSize: 1, activation: myLeakyRelu),
              ConvBN<Float>(inFilters: 512, outFilters: 1024, kernelSize: 3, activation: myLeakyRelu),
              ConvBN<Float>(inFilters: 1024, outFilters: 512, kernelSize: 1, activation: myLeakyRelu)]
    
    up = UpSampling2D<Float>(size: 2)
    convBN3 = ConvBN<Float>(inFilters: 512, outFilters: 256, kernelSize: 1, activation: myLeakyRelu)
    convBN2 = ConvBN<Float>(inFilters: 256, outFilters: 128, kernelSize: 1, activation: myLeakyRelu)
     
    
    conv_ml1 = ConvBN<Float>(inFilters: 256, outFilters: 128, kernelSize: 1, activation: myLeakyRelu)
    conv_ml2 = ConvBN<Float>(inFilters: 512, outFilters: 256, kernelSize: 1, activation: myLeakyRelu)
    conv_ml3 = ConvBN<Float>(inFilters: 128, outFilters: 256, kernelSize: 3, stride: 2, activation: myLeakyRelu)
    conv_ml4 = ConvBN<Float>(inFilters: 256, outFilters: 512, kernelSize: 3, stride: 2, activation: myLeakyRelu)
    
    oconv1 = ConvBN<Float>(inFilters: 128, outFilters: 256, kernelSize: 3, activation: myLeakyRelu)
    oconv2 = ConvBN<Float>(inFilters: 256, outFilters: 512, kernelSize: 3, activation: myLeakyRelu)
    oconv3 = ConvBN<Float>(inFilters: 512, outFilters: 1024, kernelSize: 3, activation: myLeakyRelu) 
      
    yolo1 = Conv2D<Float>(
            filterShape: (1, 1, 256, outFilters), 
            strides: (1, 1), 
            padding: .same)
    yolo2 = Conv2D<Float>(
            filterShape: (1, 1, 512, outFilters), 
            strides: (1, 1), 
            padding: .same)
    yolo3 = Conv2D<Float>(
            filterShape: (1, 1, 1024, outFilters), 
            strides: (1, 1), 
            padding: .same)
      
    
      convBNssss0 =  [ConvBN<Float>(inFilters: 512, outFilters: 256, kernelSize: 1, activation: myLeakyRelu),
              ConvBN<Float>(inFilters: 256, outFilters: 512, kernelSize: 3, activation: myLeakyRelu),
              ConvBN<Float>(inFilters: 512, outFilters: 256, kernelSize: 1, activation: myLeakyRelu),
              ConvBN<Float>(inFilters: 256, outFilters: 512, kernelSize: 3, activation: myLeakyRelu),
              ConvBN<Float>(inFilters: 512, outFilters: 256, kernelSize: 1, activation: myLeakyRelu)]
      
      convBNssss1 =  [ConvBN<Float>(inFilters: 256, outFilters: 128, kernelSize: 1, activation: myLeakyRelu),
              ConvBN<Float>(inFilters: 128, outFilters: 256, kernelSize: 3, activation: myLeakyRelu),
                     ConvBN<Float>(inFilters: 256, outFilters: 128, kernelSize: 1, activation: myLeakyRelu),
              ConvBN<Float>(inFilters: 128, outFilters: 256, kernelSize: 3, activation: myLeakyRelu),
                     ConvBN<Float>(inFilters: 256, outFilters: 128, kernelSize: 1, activation: myLeakyRelu)]
      
      convBNssss2 =  [ConvBN<Float>(inFilters: 512, outFilters: 256, kernelSize: 1, activation: myLeakyRelu),
              ConvBN<Float>(inFilters: 256, outFilters: 512, kernelSize: 3, activation: myLeakyRelu),
              ConvBN<Float>(inFilters: 512, outFilters: 256, kernelSize: 1, activation: myLeakyRelu),
              ConvBN<Float>(inFilters: 256, outFilters: 512, kernelSize: 3, activation: myLeakyRelu),
              ConvBN<Float>(inFilters: 512, outFilters: 256, kernelSize: 1, activation: myLeakyRelu)]
      
    convBNssss3 =  [ConvBN<Float>(inFilters: 1024, outFilters: 512, kernelSize: 1, activation: myLeakyRelu),
                      ConvBN<Float>(inFilters: 512, outFilters: 1024, kernelSize: 3, activation: myLeakyRelu),
                    ConvBN<Float>(inFilters: 1024, outFilters: 512, kernelSize: 1, activation: myLeakyRelu),
                      ConvBN<Float>(inFilters: 512, outFilters: 1024, kernelSize: 3, activation: myLeakyRelu),
                    ConvBN<Float>(inFilters: 1024, outFilters: 512, kernelSize: 1, activation: myLeakyRelu)]
      
      
    yoloHead1 = YOLO<Float>(imageSize: [512, 512], anchors: Tensor<Float>([[12, 16], [19, 36], [40, 28]]), cls: 80, scale: 1.2)
    yoloHead2 = YOLO<Float>(imageSize: [512, 512], anchors: Tensor<Float>([[36, 75], [76, 55], [72, 146]]), cls: 80, scale: 1.1)
    yoloHead3 = YOLO<Float>(imageSize: [512, 512], anchors: Tensor<Float>([[142, 110], [192, 243], [459, 401]]), cls: 80, scale: 1.05)
      
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
      let b = input.shape[0]
      let h = input.shape[1]/32
      let w = input.shape[2]/32                                          
      let level1 = input |> backbone.base |> backbone.cspBlock[0] |> backbone.cspBlock[1] |> backbone.cspBlock[2] //23 -> 54
      let level2 = level1 |> backbone.cspBlock[3]  // 85
      let level3 = level2 |> backbone.cspBlock[4] |> convBNs1 |> spp //114
                               
      let mlevel3 = level3 |> convBNs2
      let mlevel2 = (level2 |> conv_ml2).concatenated(with: mlevel3 |> convBN3 |> up, alongAxis: -1) |> convBNssss0
      let mlevel1 = (level1 |> conv_ml1).concatenated(with: mlevel2 |> convBN2 |> up, alongAxis: -1)
      
      let olevel1 = mlevel1 |> convBNssss1
      let olevel2 = (olevel1 |> conv_ml3).concatenated(with: mlevel2, alongAxis: -1) |> convBNssss2
      let olevel3 = (olevel2 |> conv_ml4).concatenated(with: mlevel3, alongAxis: -1) |> convBNssss3
      
      let output1 = (olevel1 |> oconv1 |> yolo1 |> yoloHead1).transposed(permutation: [0,3,1,2]).reshaped(to: [b,outFilters,h*w*16])
      let output2 = (olevel2 |> oconv2 |> yolo2 |> yoloHead2).transposed(permutation: [0,3,1,2]).reshaped(to: [b,outFilters,h*w*4])
      let output3 = (olevel3 |> oconv3 |> yolo3 |> yoloHead3).transposed(permutation: [0,3,1,2]).reshaped(to: [b,outFilters,h*w])
      
    return output1.concatenated(with: output2, alongAxis: -1).concatenated(with: output3, alongAxis: -1)
  }
}


import TensorFlow
import Foundation

public struct ConvBN<T: TensorFlowFloatingPoint>: Layer {
    public var conv: Conv2D<Float>
    public var norm: BatchNorm<Float>
    public var zero: ZeroPadding2D<Float>
    public var useZero: Bool = false
    public var activ: Activation = mish
    
    public typealias Activation = @differentiable (Tensor<Float>) -> Tensor<Float>
    public init(
        inFilters: Int,
        outFilters: Int,
        kernelSize: Int = 1,
        stride: Int = 1,
        padding: Padding = .same,
        activation: @escaping Activation = mish
    ) {
        activ = activation
        useZero = kernelSize > 1
        let padding_tmp = useZero ? .valid : padding
        conv = Conv2D(
            filterShape: (kernelSize, kernelSize, inFilters, outFilters), 
            strides: (stride, stride), 
            padding: padding_tmp)
        norm = BatchNorm(featureCount: outFilters, momentum: 0.9, epsilon: 1e-5)
        let pad  = (kernelSize-1) / 2
        zero = ZeroPadding2D<Float>(padding: (pad, pad))
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return !useZero ? activ(input |> conv |> norm) : activ(input |> zero |> conv |> norm)
    }
}


public struct ShortcutBlock<T: TensorFlowFloatingPoint>: Layer {
    public var convs: [ConvBN<T>]

    public init(inFilters: Int, outFilters: Int, middleFilters: Int){
        precondition(inFilters == outFilters, "inFilters should be equal to outFilters")
        //precondition(inFilters % 2 == 0, "inFilters should be multiples of 2")
        convs = [ConvBN<T>(inFilters: inFilters, outFilters: middleFilters, kernelSize: 1),
                 ConvBN<T>(inFilters: middleFilters, outFilters: outFilters, kernelSize: 3)]
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convResult = convs.differentiableReduce(input) { $1($0) }
        return convResult + input
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



public struct MyUpSampling2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    @noDerivative public let size: Int
      public init(size: Int) {
        self.size = size
      }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        
        let newSize = Tensor([Int32(input.shape[1]*size), Int32(input.shape[2]*size)], on: .defaultTFEager)
        var x: Tensor<Float> =  resizeNearestNeighbor(images: input, size: newSize)
        return x
        
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
      
      
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
      let b = input.shape[0]
      let h = input.shape[1]/32
      let w = input.shape[2]/32                                          
      let level1 = input |> backbone.base |> backbone.cspBlock[0] |> backbone.cspBlock[1] |> backbone.cspBlock[2] //23 -> 54
      let level2 = level1 |> backbone.cspBlock[3]  // 85
      let level3 = level2 |> backbone.cspBlock[4] |> convBNs1 |> spp 
                                  //104            
   
      let mlevel3 = level3 |> convBNs2
      let mlevel2 = (level2 |> conv_ml2).concatenated(with: mlevel3 |> convBN3 |> up, alongAxis: -1) |> convBNssss0
      let mlevel1 = (level1 |> conv_ml1).concatenated(with: mlevel2 |> convBN2 |> up, alongAxis: -1)
      
      let olevel1 = mlevel1 |> convBNssss1
      let olevel2 = (olevel1 |> conv_ml3).concatenated(with: mlevel2, alongAxis: -1) |> convBNssss2
      let olevel3 = (olevel2 |> conv_ml4).concatenated(with: mlevel3, alongAxis: -1) |> convBNssss3

      let output1 = (olevel1 |> oconv1 |> yolo1).transposed(permutation: [0,3,1,2]).reshaped(to: [b,outFilters,h*w*16])
      let output2 = (olevel2 |> oconv2 |> yolo2).transposed(permutation: [0,3,1,2]).reshaped(to: [b,outFilters,h*w*4])
      let output3 = (olevel3 |> oconv3 |> yolo3).transposed(permutation: [0,3,1,2]).reshaped(to: [b,outFilters,h*w])
    return output1.concatenated(with: output2, alongAxis: -1).concatenated(with: output3, alongAxis: -1)
  }
}


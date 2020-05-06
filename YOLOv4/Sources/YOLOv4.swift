import TensorFlow

precedencegroup ForwardFlow {
  associativity: left
}

infix operator |> : ForwardFlow

@differentiable
public func |> <Features: Differentiable, Func: Layer>(left: Func.Input, right:Func) -> Func.Output where Features == Func.Input {
  return right(left);
}

extension Array: Module where Element: Layer, Element.Input == Element.Output {
  public typealias Input = Element.Input
  public typealias Output = Element.Output

  @differentiable
  public func callAsFunction(_ input: Input) -> Output {
    differentiableReduce(input) { $1($0) }
  }
}

extension Array: Layer where Element: Layer, Element.Input == Element.Output {}


@differentiable
public func mish<T: TensorFlowFloatingPoint>(_ features: Tensor<T>) -> Tensor<T> {
  features * tanh(softplus(features))
}


public struct ConvBN<T: TensorFlowFloatingPoint>: Layer {
    public var conv: Conv2D<Float>
    public var norm: BatchNorm<Float>

    public typealias Activation = @differentiable (Tensor<Float>) -> Tensor<Float>
    public init(
        inFilters: Int,
        outFilters: Int,
        kernelSize: Int = 1,
        stride: Int = 1,
        padding: Padding = .same,
        activation: Activation = mish
    ) {
            conv = Conv2D(
            filterShape: (kernelSize, kernelSize, inFilters, outFilters), 
            strides: (stride, stride), 
            padding: padding)
            norm = BatchNorm(featureCount: outFilters, momentum: 0.9, epsilon: 1e-5)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return mish(input |> conv |> norm)
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



public struct CSPBlock<T: TensorFlowFloatingPoint>: Layer {
    public var base: ConvBN<T>
    public var part1: ConvBN<T>
    public var part2: ConvBN<T>
    public var denseBlock: [ShortcutBlock<T>]
    public var transition1: ConvBN<T>
    public var transition2: ConvBN<T>

    public init(inFilters: Int, outFilters: Int, repeatDense: Int = 1, expand: Bool = false){
      denseBlock = []
      precondition(inFilters * 2 == outFilters, "inFilters * 2 == outFilters")
      if (!expand) {   
          base  = ConvBN<T>(inFilters: inFilters, outFilters: outFilters, kernelSize: 3, stride: 2)
          part1 = ConvBN<T>(inFilters: outFilters, outFilters: inFilters, kernelSize: 1)
          part2 = ConvBN<T>(inFilters: outFilters, outFilters: inFilters, kernelSize: 1)
          for _ in 1...repeatDense {
              denseBlock.append(ShortcutBlock<T>(inFilters: inFilters, outFilters: inFilters, middleFilters: inFilters))
          }
          transition1 = ConvBN<T>(inFilters: inFilters, outFilters: inFilters, kernelSize: 1)
          transition2 = ConvBN<T>(inFilters: outFilters, outFilters: outFilters, kernelSize: 1)
      } else {
          base  = ConvBN<T>(inFilters: inFilters, outFilters: outFilters, kernelSize: 3, stride: 2)
          part1 = ConvBN<T>(inFilters: outFilters, outFilters: outFilters, kernelSize: 1)
          part2 = ConvBN<T>(inFilters: outFilters, outFilters: outFilters, kernelSize: 1)
          for _ in 1...repeatDense {
              denseBlock.append(ShortcutBlock<T>(inFilters: outFilters, outFilters: outFilters, middleFilters: inFilters))
          }
          transition1 = ConvBN<T>(inFilters: outFilters, outFilters: outFilters, kernelSize: 1)
          transition2 = ConvBN<T>(inFilters: 2*outFilters, outFilters: outFilters, kernelSize: 1)
      }
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let baseResult = input |> base
        let route1Result = baseResult |> part1
        let route2Result = baseResult |> part2 |> denseBlock |> transition1
        return route1Result.concatenated(with: route2Result, alongAxis: 3) |> transition2
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


public struct SPP<T: TensorFlowFloatingPoint>: Layer {
    public var maxpoolHigh: MaxPool2D<Float>
    public var maxpoolMid: MaxPool2D<Float>
    public var maxpoolLow: MaxPool2D<Float>
    public init(){
        maxpoolHigh = MaxPool2D<Float>(poolSize: (1, 5, 5, 1), strides:  (1,1,1,1), padding: .same)//2
        maxpoolMid = MaxPool2D<Float>(poolSize: (1, 9, 9, 1), strides:  (1,1,1,1), padding: .same)//4
        maxpoolLow = MaxPool2D<Float>(poolSize: (1, 13, 13, 1), strides:  (1,1,1,1), padding: .same)//6
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let high = input |> maxpoolHigh
        let mid  = input |> maxpoolMid
        let low  = input |> maxpoolLow
        return high.concatenated(with: mid, alongAxis: -1).concatenated(with: low, alongAxis: -1).concatenated(with: input, alongAxis: -1)
    }
}


@differentiable(wrt: x)
public func leakyRelu<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
    leakyRelu(x, alpha: 0.1)
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
    convBNs1 = [ConvBN<Float>(inFilters: 1024, outFilters: 512, kernelSize: 1, activation: leakyRelu),
              ConvBN<Float>(inFilters: 512, outFilters: 1024, kernelSize: 3, activation: leakyRelu),
              ConvBN<Float>(inFilters: 1024, outFilters: 512, kernelSize: 1, activation: leakyRelu)]
      
   convBNs2 = [ConvBN<Float>(inFilters: 2048, outFilters: 512, kernelSize: 1, activation: leakyRelu),
              ConvBN<Float>(inFilters: 512, outFilters: 1024, kernelSize: 3, activation: leakyRelu),
              ConvBN<Float>(inFilters: 1024, outFilters: 512, kernelSize: 1, activation: leakyRelu)]
    
    up = UpSampling2D<Float>(size: 2)
    convBN3 = ConvBN<Float>(inFilters: 512, outFilters: 256, kernelSize: 1, activation: leakyRelu)
    convBN2 = ConvBN<Float>(inFilters: 256, outFilters: 128, kernelSize: 1, activation: leakyRelu)
      
    conv_ml1 = ConvBN<Float>(inFilters: 256, outFilters: 256, kernelSize: 1, activation: leakyRelu)
    conv_ml2 = ConvBN<Float>(inFilters: 512, outFilters: 512, kernelSize: 1, activation: leakyRelu)
    conv_ml3 = ConvBN<Float>(inFilters: 128, outFilters: 256, kernelSize: 1, stride: 2, activation: leakyRelu)
    conv_ml4 = ConvBN<Float>(inFilters: 256, outFilters: 512, kernelSize: 1, stride: 2, activation: leakyRelu)
    
    oconv1 = ConvBN<Float>(inFilters: 128, outFilters: 256, kernelSize: 3, activation: leakyRelu)
    oconv2 = ConvBN<Float>(inFilters: 256, outFilters: 512, kernelSize: 3, activation: leakyRelu)
    oconv3 = ConvBN<Float>(inFilters: 512, outFilters: 1024, kernelSize: 3, activation: leakyRelu) 
      
    yolo1 = Conv2D<Float>(
            filterShape: (3, 3, 256, outFilters), 
            strides: (1, 1), 
            padding: .same)
    yolo2 = Conv2D<Float>(
            filterShape: (3, 3, 512, outFilters), 
            strides: (1, 1), 
            padding: .same)
    yolo3 = Conv2D<Float>(
            filterShape: (3, 3, 1024, outFilters), 
            strides: (1, 1), 
            padding: .same)
      
    
      convBNssss0 =  [ConvBN<Float>(inFilters: 768, outFilters: 256, kernelSize: 1, activation: leakyRelu),
              ConvBN<Float>(inFilters: 256, outFilters: 512, kernelSize: 3, activation: leakyRelu),
              ConvBN<Float>(inFilters: 512, outFilters: 256, kernelSize: 1, activation: leakyRelu),
              ConvBN<Float>(inFilters: 256, outFilters: 512, kernelSize: 3, activation: leakyRelu),
              ConvBN<Float>(inFilters: 512, outFilters: 256, kernelSize: 1, activation: leakyRelu)]
      
      convBNssss1 =  [ConvBN<Float>(inFilters: 384, outFilters: 128, kernelSize: 1, activation: leakyRelu),
              ConvBN<Float>(inFilters: 128, outFilters: 256, kernelSize: 3, activation: leakyRelu),
                     ConvBN<Float>(inFilters: 256, outFilters: 128, kernelSize: 1, activation: leakyRelu),
              ConvBN<Float>(inFilters: 128, outFilters: 256, kernelSize: 3, activation: leakyRelu),
                     ConvBN<Float>(inFilters: 256, outFilters: 128, kernelSize: 1, activation: leakyRelu)]
      
      convBNssss2 =  [ConvBN<Float>(inFilters: 512, outFilters: 256, kernelSize: 1, activation: leakyRelu),
              ConvBN<Float>(inFilters: 256, outFilters: 512, kernelSize: 3, activation: leakyRelu),
              ConvBN<Float>(inFilters: 512, outFilters: 256, kernelSize: 1, activation: leakyRelu),
              ConvBN<Float>(inFilters: 256, outFilters: 512, kernelSize: 3, activation: leakyRelu),
              ConvBN<Float>(inFilters: 512, outFilters: 256, kernelSize: 1, activation: leakyRelu)]
      
    convBNssss3 =  [ConvBN<Float>(inFilters: 1024, outFilters: 512, kernelSize: 1, activation: leakyRelu),
                      ConvBN<Float>(inFilters: 512, outFilters: 1024, kernelSize: 3, activation: leakyRelu),
                    ConvBN<Float>(inFilters: 1024, outFilters: 512, kernelSize: 1, activation: leakyRelu),
                      ConvBN<Float>(inFilters: 512, outFilters: 1024, kernelSize: 3, activation: leakyRelu),
                    ConvBN<Float>(inFilters: 1024, outFilters: 512, kernelSize: 1, activation: leakyRelu)]
      
      
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
      let b = input.shape[0]
      let h = input.shape[1]/32
      let w = input.shape[2]/32
      let level1 = input |> backbone.base |> backbone.cspBlock[0] |> backbone.cspBlock[1] |> backbone.cspBlock[2] //54
      let level2 = level1 |> backbone.cspBlock[3]  //85
      let level3 = level2 |> backbone.cspBlock[4] |> convBNs1 |> spp
      
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

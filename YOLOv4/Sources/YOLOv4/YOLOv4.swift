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
                           CSPBlock<T>(inFilters: 128, outFilters: 256, repeatDense: 8),
                           CSPBlock<T>(inFilters: 256, outFilters: 512, repeatDense: 8),
                           CSPBlock<T>(inFilters: 512, outFilters: 1024, repeatDense: 4)]
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
        maxpoolHigh = MaxPool2D<Float>(poolSize: (1, 5, 5, 1), strides:  (1,1,1,1), padding: .same)
        maxpoolMid = MaxPool2D<Float>(poolSize: (1, 9, 9, 1), strides:  (1,1,1,1), padding: .same)
        maxpoolLow = MaxPool2D<Float>(poolSize: (1, 13, 13, 1), strides:  (1,1,1,1), padding: .same)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return   (input |> maxpoolHigh)
               + (input |> maxpoolMid)
               + (input |> maxpoolLow)
               + input
    }
}

import TensorFlow
import Foundation

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


@differentiable(wrt: x)
public func myLeakyRelu<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
    leakyRelu(x, alpha: 0.1)
}

@differentiable
public func mish<T: TensorFlowFloatingPoint>(_ features: Tensor<T>) -> Tensor<T> {
  features * tanh(softplus(features))
}

public func getWeights <T: TensorFlowFloatingPoint> (
    _ stream: InputStream,
    _ size: Int) -> Tensor<T> {
            var buf:[UInt8] = [UInt8](repeating: 0, count:  size * MemoryLayout<T>.size)
            var f: [T] = [T](repeating: 0, count: size)
            var len = stream.read(&buf, maxLength: buf.count)
            memcpy(&f, buf, len)
            var weights = Tensor<T>(shape:[size], scalars: f)
    return weights
}

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

    public mutating func load(from stream: InputStream) {
        let shape = conv.filter.shape 
        let yolo_shape = TensorShape([shape[3], shape[2], shape[0], shape[1]])
        let yolo2TF = [2, 3, 1, 0]
        norm.offset = getWeights(stream, shape[3])
        norm.scale = getWeights(stream, shape[3])
        norm.runningMean = Parameter(getWeights(stream, shape[3]))
        norm.runningVariance = Parameter(getWeights(stream, shape[3]))
        conv.filter = getWeights(stream, shape.contiguousSize)
                        .reshaped(to: yolo_shape)
                        .transposed(permutation: yolo2TF)
    }
    
    public func load(from stream: InputStream) -> ConvBN {
        var x = self
        let shape = conv.filter.shape 
        let yolo_shape = TensorShape([shape[3], shape[2], shape[0], shape[1]])
        let yolo2TF = [2, 3, 1, 0]
        x.norm.offset = getWeights(stream, shape[3])
        x.norm.scale = getWeights(stream, shape[3])
        x.norm.runningMean = Parameter(getWeights(stream, shape[3]))
        x.norm.runningVariance = Parameter(getWeights(stream, shape[3]))
        x.conv.filter = getWeights(stream, shape.contiguousSize)
                        .reshaped(to: yolo_shape)
                        .transposed(permutation: yolo2TF)
        return x
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
    
     public mutating func load(from stream: InputStream) {
           convs = convs.map{$0.load(from: stream)}
    }
    public func load(from stream: InputStream) -> ShortcutBlock {
           var x = self
           x.convs = convs.map{$0.load(from: stream)}
            return x
    }
}

extension Conv2D {
        public mutating func load(from stream: InputStream) {
        let shape = filter.shape 
        let yolo_shape = TensorShape([shape[3], shape[2], shape[0], shape[1]])
        let yolo2TF = [2, 3, 1, 0]
        bias = getWeights(stream, shape[3])
        filter = getWeights(stream, shape.contiguousSize)
                        .reshaped(to: yolo_shape)
                        .transposed(permutation: yolo2TF)
    }
    
   public func load(from stream: InputStream) -> Conv2D {
        var x = self
        let shape = filter.shape 
        let yolo_shape = TensorShape([shape[3], shape[2], shape[0], shape[1]])
        let yolo2TF = [2, 3, 1, 0]
        x.bias = getWeights(stream, shape[3])
        x.filter = getWeights(stream, shape.contiguousSize)
                        .reshaped(to: yolo_shape)
                        .transposed(permutation: yolo2TF)
       return x
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
   /* 
    public mutating func load(from stream: InputStream) {
        base.load(from: stream)
        part1.load(from: stream)
        part2.load(from: stream)
        denseBlock = denseBlock.map{$0.load(from: stream)}
        transition1.load(from: stream)
        transition2.load(from: stream)
    }*/ 
    
    public func load(from stream: InputStream) -> CSPBlock {
        var x = self
        x.base = base.load(from: stream)
        x.part1 = part1.load(from: stream)
        x.part2 = part2.load(from: stream)
        x.denseBlock = denseBlock.map{$0.load(from: stream)}
        x.transition1 = transition1.load(from: stream)
        x.transition2 = transition2.load(from: stream)
        return x
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let baseResult = input |> base
        let route1Result = baseResult |> part1
        let route2Result = baseResult |> part2 |> denseBlock |> transition1
        let concate = route2Result.concatenated(with: route1Result, alongAxis: 3)
        let output = concate |> transition2
        return output
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
    /*
   public mutating func load(from stream: InputStream) {
       base.load(from: stream)
       cspBlock.load(from: stream)
   }*/
    
   public func load(from stream: InputStream) -> CSPDarknet53 {
       var x = self
       x.base = base.load(from: stream)
       x.cspBlock = cspBlock.map{$0.load(from: stream)}
       return x
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
    /*
   public mutating func load(from stream: InputStream) {
       base.load(from: stream)
       conv.load(from: stream)
   }*/
    
   public func load(from stream: InputStream) -> CSPDarknet53ImageNet {
       var x = self
       x.base = base.load(from: stream)
       x.conv = conv.load(from: stream)
       return x
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


@usableFromInline
@differentiable(wrt: images where Scalar: TensorFlowFloatingPoint)
func resizeNearestNeighbor<Scalar: TensorFlowNumeric>(
  images: Tensor<Scalar>,
  size: Tensor<Int32>,
  alignCorners: Bool = true,
  halfPixelCenters: Bool = false
) -> Tensor<Scalar> {
  _Raw.resizeNearestNeighbor(
    images: images,
    size: size,
    alignCorners: alignCorners,
    halfPixelCenters: halfPixelCenters
  )
}

@usableFromInline
@derivative(of: resizeNearestNeighbor)
func _vjpResizeNearestNeighbor<Scalar: TensorFlowFloatingPoint>(
  images: Tensor<Scalar>,
  size: Tensor<Int32>,
  alignCorners: Bool,
  halfPixelCenters: Bool
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> Tensor<Scalar>) {
  let resized = resizeNearestNeighbor(
    images: images,
    size: size,
    alignCorners: alignCorners,
    halfPixelCenters: halfPixelCenters
  )
  return (
    resized,
    { v in
      _Raw.resizeNearestNeighborGrad(
        grads: v,
        size: Tensor([Int32(images.shape[1]), Int32(images.shape[2])], on: .defaultTFEager),
        alignCorners: alignCorners,
        halfPixelCenters: halfPixelCenters
      )
    }
  )
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
  public var up: MyUpSampling2D<Float>
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
    
    up = MyUpSampling2D<Float>(size: 2)
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
          print(1)                        //104            
   
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
   /*
    public mutating func load(from stream: InputStream) {
          backbone.load(from: stream)
          convBNs1.load(from: stream)
          convBNs2.load(from: stream)
          conv_ml2.load(from: stream)
          convBN3.load(from: stream)
          convBNssss0.load(from: stream)
          conv_ml1.load(from: stream)
          convBN2.load(from: stream)
        
          convBNssss1.load(from: stream)
          conv_ml3.load(from: stream)
          convBNssss2.load(from: stream)
          conv_ml4.load(from: stream)
          convBNssss3.load(from: stream)
          oconv1.load(from: stream)
          oconv2.load(from: stream)
          oconv3.load(from: stream)
          yolo1.load(from: stream)
          yolo2.load(from: stream)
          yolo3.load(from: stream)
   }*/
    
   public func load(from stream: InputStream) -> YOLOv4 {
       var x = self
           x.backbone = backbone.load(from: stream)
          x.convBNs1 = convBNs1.map{$0.load(from: stream)}
          x.convBNs2 = convBNs2.map{$0.load(from: stream)}
          
          x.convBN3 = convBN3.load(from: stream)
          x.conv_ml2 = conv_ml2.load(from: stream)
          
          x.convBNssss0 = convBNssss0.map{$0.load(from: stream)}
       
          x.convBN2 = convBN2.load(from: stream)
          x.conv_ml1 = conv_ml1.load(from: stream)
       
          x.convBNssss1 = convBNssss1.map{$0.load(from: stream)}
       x.oconv1 = oconv1.load(from: stream)   
       x.yolo1 = yolo1.load(from: stream)
          x.conv_ml3 = conv_ml3.load(from: stream)
          x.convBNssss2 = convBNssss2.map{$0.load(from: stream)}
       x.oconv2 = oconv2.load(from: stream)   
       x.yolo2 = yolo2.load(from: stream)
          x.conv_ml4 = conv_ml4.load(from: stream)
          x.convBNssss3 = convBNssss3.map{$0.load(from: stream)}
          
          
          x.oconv3 = oconv3.load(from: stream)
          x.yolo3 = yolo3.load(from: stream)
       return x
   }
   
}


import Dispatch
public func time(_ function: () -> ()) {
    let start = DispatchTime.now()
    function()
    let end = DispatchTime.now()
    let nanoseconds = Double(end.uptimeNanoseconds - start.uptimeNanoseconds)
    let milliseconds = nanoseconds / 1e6
    // Could do fancier pretty-printing logic based on the order of magnitude.
    print("\(milliseconds) ms")
}

public func time(re: Int, _ function: () -> ()) {
    time{
        for i in 1...re {
            function()
                        }
        }
  // Run it a few times and calculate mean and stdev.
}

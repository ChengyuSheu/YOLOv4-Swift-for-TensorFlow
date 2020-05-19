import TensorFlow

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

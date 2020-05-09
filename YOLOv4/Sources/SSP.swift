import TensorFlow

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

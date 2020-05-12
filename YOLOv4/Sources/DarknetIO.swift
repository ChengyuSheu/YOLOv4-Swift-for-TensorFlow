import TensorFlow
import Foundation

extension CSPBlock {
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
}

/*
extension Array {
    public func load(from stream: InputStream) -> Array {
        var x = self
        x.map{$0.load(from: stream)}
        return x
    }
}*/

extension ConvBN {
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
}

extension ShortcutBlock {
    public func load(from stream: InputStream) -> ShortcutBlock {
           var x = self
           x.convs = convs.map{$0.load(from: stream)}
            return x
    }
}

extension Conv2D {
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

extension CSPDarknet53 {
   public func load(from stream: InputStream) -> CSPDarknet53 {
       var x = self
       x.base = base.load(from: stream)
       x.cspBlock = cspBlock.map{$0.load(from: stream)}
       return x
   }
}

extension CSPDarknet53ImageNet {
   public func load(from stream: InputStream) -> CSPDarknet53ImageNet {
       var x = self
       x.base = base.load(from: stream)
       x.conv = conv.load(from: stream)
       return x
   }
}

extension YOLOv4 {
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
    
    
    public func load(fileAtPath path: String, ignoreBytes: Int = 20) -> YOLOv4 {
        var x = self
        if let stream:InputStream = InputStream(fileAtPath: path) {
            stream.open()
            var head:[UInt8] = [UInt8](repeating: 0, count: ignoreBytes)
            stream.read(&head, maxLength: head.count)
            x = self.load(from: stream)
            stream.close()
        }
        return x
    }
}
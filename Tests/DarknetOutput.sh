git clone https://github.com/AlexeyAB/darknet.git

ln -s dog_512.jpg darknet/dog_512.jpg

cd darknet

# download model
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

#reference version, 2020.5.18
git checkout 594f7ce6aff73c3e7f55b6fc31ee9a13feb5e4ef

#inject feature output function for the yolo layer
match='int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets, letter);'
insert='FILE* fp = fopen("output.dat", "a");
            if (fp) {
                printf("outputs: %d\n", l.outputs);
                fwrite(&l.outputs, sizeof(int), 1, fp);
                fwrite(l.output, sizeof(float), l.outputs, fp);
                fclose(fp);
            }'
file='src/network.c'

#inplace  "s/ old string / new string /"  target
sed -i "s/$match/$insert\n$match/" $file


make
./darknet detector test ./cfg/coco.data ./cfg/yolov4.cfg ./yolov4.weights dog_512.jpg -i 0 -thresh 0.25

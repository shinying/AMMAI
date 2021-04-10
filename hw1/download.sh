echo "Download RetinaFace pretrained weights"
wget http://cmlab.csie.ntu.edu.tw/~shinlee/retina_weights.zip
unzip retina_weights.zip
mv Resnet50_Final.pth retinaface/weights
mv Resnet50_ours.pth retinaface/weights
rm retina_weights.zip

echo "Download face verification weights"

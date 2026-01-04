accelerate launch train.py --config configs/irepa.yaml \
  --model="SiT-B/2" \
  --enc-type="dinov2-vit-b" \
  --encoder-depth=6 \
  --data-dir=/home/jiacheng/imagenet_repa \
  --exp-name="sitb2-pe-vit-g-irepa"
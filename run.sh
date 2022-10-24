git clone https://github.com/rwightman/pytorch-image-models.git -b 0.5.x
cd pytorch-image-models && python setup.py install
cd ..

python main.py --eval --resume /home2/pytorch-broad-models/deitb/model/deit_base_patch16_224-b5f2ef4d.pth --data-path /home2/pytorch-broad-models/imagenet --device xpu --precision float16 --jit --channels_last 1 --num_iter 200 --num_warmup 20 --batch_size 1

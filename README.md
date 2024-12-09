## use 4 type of images to predict


## install pytorch
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Pre-process data
```bash
python pre_process_data.py --raw_dir ../data/raw/700hPa-train --output_dir ../data/train --prefixes wv vor wind z
python pre_process_data.py --raw_dir ../data/raw/700hPa-test --output_dir ../data/test --prefixes wv vor wind z
```

## Process data
```bash
python process_data.py --subsets wv vor wind z  --num_clips 5000000
```

## local training
```bash
python avg_runner.py -n code4run01 -s 1000000 --test_freq 200000 -ns 4 -td ../data/clips/wv-vor-wind-z/ -t ../data/test/wv/ ../data/test/vor/ ../data/test/wind/ ../data/test/z/
```

## local test
```bash
python avg_runner.py -n code2run01 -b 1 -ns 4 -t ../data/test/wv/ ../data/test/vor/ ../data/test/wind/ ../data/test/z/ -T
```

## batch test
```bash
python avg_runner.py -n code2run01 -b 1 -ns 4 -t ../data/test/wv/ ../data/test/vor/ ../data/test/wind/ ../data/test/z/ -d seperate-figs-test -BT
```


## batch test
```bash
python avg_runner2.py -n code5run01 -b 1 -ns 4 -t ../data/test/wv/ ../data/test/vor/ ../data/test/wind/ ../data/test/z/ -d 2015045S12145-LAM -BT
python avg_runner2.py -n code5run01 -b 1 -ns 4 -t ../data/test/wv/ ../data/test/vor/ ../data/test/wind/ ../data/test/z/ -d 2020006S16121-BLAKE -BT
python avg_runner2.py -n code5run01 -b 1 -ns 4 -t ../data/test/wv/ ../data/test/vor/ ../data/test/wind/ ../data/test/z/ -d 2020037S17121-DAMIEN -BT
```

## view clip
```bash
python view_clip.py --folder ../data/clips/vor-wv/ --file 0 1 2 3 4
```


## export local conda env
```bash
conda env export --no-builds | grep -v "^prefix: " > environment.yml
```

## watch nvidia-smi
```bash
watch -n 1 nvidia-smi
```


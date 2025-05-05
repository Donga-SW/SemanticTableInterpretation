# KorSTI



이 모델은 의미추론, 개체연결, 관계주석, 삼항 구조를 통한 지식그래프 구축 모델입니다.


![STI_diagram](https://github.com/hoonZeee/STI/blob/main/STI_diagram.jpg)


## STI 개발 진행 링크


- [Jihoon_STI_Research](https://github.com/hoonZeee/STI)


## Directory

```console
KorSTI/
├── data/
│   ├── trained_mixed_name.csv
│   ├── trained_mixed_activist.csv
│   ├── trained_mixed_year.csv
│   ├── trained_mixed_judgement.csv
│   ├── trained_mixed_event.csv
│   ├── informal_file.csv
│   └── formal.csv
│
├── model/
│   ├── pretrainedModel.pt
│   └── custom_trained.pt
│
├── notebooks/
│   └── training_KoBERT.ipynb
│
├── scripts/
│   ├── extract.py
│   ├── CTA_trained.py
│   ├── CEA.py
│   ├── triple_generator.py
│
├── outputs/
│   ├── CTA_predict.csv
│   ├── CEA_result.csv
│   └── triple_output.csv
│
├── knowledge_graph/
│
└── README.md
```

## Installation

```console
$ git clone [link to repo]
$ cd korsti
$ pip install -r requirements.txt
```


## CTA Data Preparation


```console
trained_mixed_name.csv
trained_mixed_activist.csv
trained_mixed_year.csv
trained_mixed_judgement.csv
trained_mixed_event.csv
```



## CTA Training



```console
optional arguments:
  -h, --help            show this help message and exit
  --input_csv INPUT_CSV
                        학습에 사용할 CSV 파일 경로 (예: train_mixed_name.csv)
  --encoding ENCODING   입력 CSV 파일의 인코딩 방식 (기본값: utf-8)
  --model_name MODEL_NAME
                        허깅페이스에서 불러올 사전학습 KoBERT 모델 이름 (기본값: monologg/kobert)
  --max_length MAX_LENGTH
                        입력 시퀀스 최대 길이 (기본값: 32)
  --batch_size BATCH_SIZE
                        학습 배치 사이즈 (기본값: 32)
  --epoch EPOCH         학습 에폭 수 (기본값: 7) - 최적 사이즈입니다.
  --learning_rate LR    학습률 (기본값: 5e-5)
  --num_classes NUM_CLASSES
                        분류할 클래스 수 (자동으로 라벨 수에 따라 결정됨)
  --device DEVICE       학습 디바이스 설정: 'cuda' or 'cpu' (기본 자동 감지)
  --save_path SAVE_PATH
                        학습된 모델 저장 경로 (기본값: kobert_name_finetuned.pt)
```


## CTA prediction

### Usage

```console
$ python predict_name_column.py
```



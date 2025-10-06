# KorSTI


이 모델은 의미추론, 개체연결, 관계주석, 삼항 구조를 통한 지식그래프 구축 모델입니다.



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


## Text_ Embedding

- 파인튜닝된 pt파일을 활용하여 text_embedding 예측수행


```console
$ python predict_name_column.py
```

# Knowledge Graph Embedding with TransE

## 1. Overview

이 프로젝트는 독립운동 관련 데이터(인물, 사건, 죄명, 지역, 연도)를 기반으로 지식그래프(Knowledge Graph)를 구축하고, TransE 임베딩 모델을 통해 노드 간의 의미적 유사성(Semantic Similarity)을 학습한 뒤 그 결과를 2차원 공간에 시각화하여 분석하는 것을 목표로 한다.

---

## 2. Motivation

기존의 지식그래프는

> 인물 → 사건, 사건 → 죄명
> 처럼 명시적 관계로 연결된 “정적 시각화”에 머물러 있었다.

이 접근은 사용자가 그래프를 직접 탐색해야 의미를 파악할 수 있고, 잠재적 관계(implicit relation)를 자동으로 파악하기 어렵다는 한계가 있다.

이를 극복하기 위해 우리는 TransE 기반 임베딩 학습을 통해 그래프의 구조를 벡터 공간(Vector Space)으로 투영하여,

> “노드 간의 의미적 관계를 수치적으로 계산할 수 있는 기반”
> 을 구축하였다.

---

## 3. Data Preprocessing

원본 CSV 데이터 예시:

| 성명       | 성명ID  | 죄명    | 사건_ID | 주소 | 판결날짜    | ... |
| -------- | ----- | ----- | ----- | -- | ------- | --- |
| 김승한(金承翰) | N0001 | 출판법위반 | e001  | 경성 | 1912-05 | ... |

이를 아래와 같은 (head, relation, tail) 형태의 트리플(Triplet)로 변환한다.

| head  | relation | tail       |
| ----- | -------- | ---------- |
| N0001 | 죄명       | 출판법위반      |
| N0001 | 판결연도     | year_1912  |
| N0001 | 연루사건     | e001       |
| 출판법위반 | URI      | Q134434680 |
| 경성    | URI      | Q483984    |

이 데이터는 지식그래프의 기본 단위인 3요소 구조 (Head Entity, Relation, Tail Entity)를 따르며, 모든 지식은 이 트리플 집합으로 표현된다.

---

## 4. TransE: Model Concept

TransE (Translating Embeddings for Modeling Multi-relational Data) 은 Facebook AI Research에서 제안된 대표적인 관계 임베딩 모델이다.

핵심 아이디어는 단순하다:

> “하나의 관계는 벡터 공간에서의 평행 이동(translation)으로 표현될 수 있다.”

즉, 관계 ( r )이 존재할 때 Head ( h )와 Tail ( t ) 사이에는 다음 관계가 성립해야 한다:

[
\mathbf{h} + \mathbf{r} \approx \mathbf{t}
]

---

## 5. Objective Function

TransE는 위 식을 만족하도록 모든 임베딩을 학습한다.
구체적으로는, “올바른(truth) 삼중항은 가깝게”, “잘못된(negatively sampled) 삼중항은 멀게” 만드는 ranking loss를 최소화한다.

[
L = \sum_{(h, r, t) \in S} \sum_{(h', r, t') \in S'} \max(0, \gamma + d(h + r, t) - d(h' + r, t'))
]

여기서

* ( d(\cdot) ) : L1 또는 L2 거리 함수
* ( \gamma ) : margin
* ( S ) : 실제 트리플 집합
* ( S' ) : negative sample (랜덤으로 꼬리 엔티티 바꾼 것)

즉,

> 진짜 관계는 가깝게,
> 가짜 관계는 멀게 학습하도록
> 네트워크를 훈련한다.

---

## 6. Implementation

### Step 1. Triplet 변환

```python
df = pd.read_csv("../data/independence-cases.csv")
triplets = [(row["성명ID"], "죄명", row["죄명"]) for _, row in df.iterrows()]
```

### Step 2. TransE 모델 구현

```python
def distance(h, r, t):
    return torch.norm(h + r - t, p=1, dim=1)

def transe_loss(pos, neg, margin):
    return torch.relu(margin + distance(*pos) - distance(*neg)).mean()
```

### Step 3. 학습

```python
for epoch in range(EPOCHS):
    pos = (entity_emb(h), relation_emb(r), entity_emb(t))
    neg = (entity_emb(h), relation_emb(r), entity_emb(t_neg))
    loss = transe_loss(pos, neg, margin)
    loss.backward()
```

결과적으로 각 엔티티와 관계는 128차원 벡터 공간 상의 점으로 표현된다.

---

## 7. Visualization

학습된 벡터는 차원이 높으므로, t-SNE를 사용해 2차원으로 축소하여 시각화했다.

```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=50, random_state=42)
entity_2d = tsne.fit_transform(entity_emb.weight.detach().cpu().numpy())
```

* 보라색: Person
* 초록색: Event
* 하늘색: Crime
* 주황색: Address
* 노란색: Year

이후 Matplotlib으로 시각화:

```python
plt.scatter(xs, ys, c=colors, s=10)
```

---

## 8. Result Interpretation

이 결과는 단순한 시각화가 아니라, 모델이 그래프 내의 관계를 “의미적으로 이해하고 있다”는 증거다.

예를 들어:

| 의미적 패턴              | 벡터 공간에서의 현상                    |
| ------------------- | ------------------------------ |
| “보안법위반”, “치안유지법위반”  | 같은 클러스터 내 근접 배치                |
| “강원도”, “양양군”, “화천군” | 동일 지역 기반의 인물 노드 근처에 위치         |
| “year_1919”         | 3·1운동 관련 인물/사건 주변에 집중          |
| “year_1938”         | 태백산 기도제·선도교 결사 사건 관련 인물 근처에 위치 |

즉, TransE는 노드 간의 구조적, 의미적 유사성을 학습해 “서로 관련된 노드들을 공간적으로 가깝게 배치”한 것이다.

---

## 9. Conclusion

| Before     | After            |
| ---------- | ---------------- |
| 명시적 연결만 가능 | 잠재적 관계까지 추론 가능   |
| 시각적 탐색 중심  | 수치 기반 의미 분석 가능   |
| 정적 지식그래프   | 임베딩 기반 동적 그래프 확장 |

이 모델은 이후

* 유사 인물 추천,
* 누락된 관계 예측(Link Prediction),
* 자연어 질의에 대한 의미 기반 검색
  등의 고도화된 지식그래프 응용의 핵심 기반이 된다.

---

## 10. References

* Bordes et al. (2013). Translating Embeddings for Modeling Multi-relational Data.
  Advances in Neural Information Processing Systems (NeurIPS)
* PyTorch Documentation: [https://pytorch.org](https://pytorch.org)
* scikit-learn t-SNE: [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)


# AI-Semantle 분석 보고서

## 1. 문제 정의: Semantle 게임

Semantle은 숨겨진 비밀 단어를 추측하는 게임입니다. 플레이어는 단어를 추측할 때마다 비밀 단어와의 **의미적 유사도**에 기반한 점수를 피드백으로 받습니다. 점수가 높을수록 추측한 단어가 비밀 단어와 의미적으로 가깝다는 뜻입니다. AI는 이 유사도 점수 피드백을 활용하여 제한된 횟수 안에 비밀 단어를 효율적으로 찾아야 합니다. 주요 과제는 방대한 영어 단어 중에서 어떤 단어를 추측해야 가장 효과적으로 정답에 접근할 수 있을지 결정하는 것입니다.

## 2. AI 접근 방식 개요

이 프로젝트는 Semantle 게임을 해결하기 위해 다음과 같은 주요 구성 요소를 결합한 강화학습 기반 접근 방식을 사용합니다:

*   **Q-러닝 (Q-Learning)**: 어떤 **단어 클러스터**에서 다음 단어를 추측하는 것이 장기적으로 가장 높은 보상(유사도 점수)을 가져올지 학습하는 핵심 의사결정 알고리즘입니다. (`src/qlearning.py`, `src/qlearning.ipynb` 등에서 구현됨)
*   **단어 클러스터링 (Word Clustering)**: 방대한 영어 단어들을 의미적 유사성에 따라 그룹화(클러스터링)하여 AI가 탐색해야 할 단어 공간의 복잡성을 줄입니다. (결과는 `src/clusters.py`, `src/secondLevelClusters.py` 등에 저장됨)
*   **휴리스틱 함수 (Heuristic Function)**: Q-러닝 에이전트가 특정 클러스터를 선택하면, 그 클러스터 내에서 실제로 추측할 가장 유망한 단어를 선택하는 규칙 기반 또는 모델 기반의 함수입니다. (`src/qlearning.py`, `src/heuristicrnn.py`, `Testing/heuristicAlternatives.py` 등에 구현됨)
*   **계층적 Q-에이전트 구조 (README 언급 및 `qlearning.ipynb` 구현 시도)**: README에서는 문제의 복잡성을 더 줄이기 위해 메인 Q-에이전트가 하위 Q-에이전트(각각 특정 클러스터 그룹 담당)를 선택하는 계층적 구조를 언급합니다. `src/secondLevelClusters.py` 파일은 계층적 클러스터 데이터를 포함하며, **`src/qlearning.ipynb`** 에서는 이 데이터를 로드하고 **계층적 상태(`state`, `sub_state`) 및 관련 로직을 구현**하려는 명확한 시도가 확인됩니다. 반면, `qlearning.py`에는 단일 Q-에이전트만 구현되어 있습니다.

## 3. 세부 구성 요소 분석

### 3.1. 단어 임베딩 및 유사도 계산

*   **Word2Vec (Google News Dataset)**: 단어의 의미를 벡터 공간에 표현하기 위해 사전 훈련된 Google News Word2Vec 모델을 사용합니다 (`gensim` 라이브러리). `qlearning.py`, `Testing/clusterAl.py`, `Testing/Semantle.py` 등 여러 파일에서 모델을 로드하여 사용합니다.
*   **유사도 계산**: 두 단어 벡터 간의 코사인 유사도를 계산하여 의미적 유사성을 정량화합니다 (`gnews_model.similarity(word1, word2)`). 이 값은 0에서 1 사이이며, 100을 곱하여 Semantle 게임의 점수처럼 사용됩니다 (`qlearning.py`의 `guess_word` 함수, `Testing/Semantle.py`의 게임 로직). 이것이 AI가 받는 핵심 피드백입니다.

### 3.2. 단어 클러스터링

*   **목적**: 전체 단어 사전을 직접 탐색하는 것은 매우 비효율적이므로, Word2Vec 벡터 공간에서 의미적으로 유사한 단어들을 그룹(클러스터)으로 묶어 탐색 공간을 효과적으로 축소합니다. AI는 개별 단어 대신 클러스터 단위로 탐색 및 학습 전략을 세웁니다.
*   **방법 및 관련 파일**:
    *   **클러스터링 알고리즘**: 이 프로젝트에서는 최소 두 가지 클러스터링 알고리즘이 사용되었습니다:
        *   **`sklearn.cluster.AgglomerativeClustering` (응집형 계층적 클러스터링)**: README에서 언급되었고 `Testing/clusterAl.py`에서 확인된 방식으로, '상향식'으로 유사한 클러스터를 병합합니다. (`n_clusters=8`, `metric='euclidean'`, `linkage='ward'` 파라미터 사용 예시 확인).
        *   **`sklearn.cluster.KMeans` (K-평균 클러스터링)**: `src/cluster.ipynb`에서 사용된 방식으로, 미리 지정된 클러스터 수(k=8)를 기준으로 중심점을 찾아 데이터를 그룹화합니다.
    *   **입력 데이터**: 분석된 코드(`cluster.ipynb`, `Testing/clusterAl.py`) 모두 클러스터링의 입력으로 `answers.py`에 정의된 **정답 단어 목록(`secretWords`)의 Word2Vec 임베딩 벡터**를 사용합니다. 즉, 가능한 정답 단어들을 기준으로 클러스터링을 수행합니다.
    *   **결과 저장**:
        *   `src/clusters.py`: 클러스터링 결과를 **리스트의 리스트 (`list[list[str]]`)** 형태로 저장합니다. 각 내부 리스트는 하나의 클러스터에 속한 단어 목록입니다. `qlearning.py`는 이 파일을 로드하여 사용합니다.
        *   `src/secondLevelClusters.py`: 클러스터링 결과를 **리스트의 리스트의 리스트 (`list[list[list[str]]]`)** 라는 더 복잡한 중첩 구조로 저장합니다. 이는 계층적 또는 다단계 클러스터링 결과 데이터이며, **`src/qlearning.ipynb`에서 계층적 에이전트 구현에 사용됩니다.**
    *   **시각화**: `src/cluster.ipynb`에서는 UMAP을 이용한 차원 축소와 Plotly 라이브러리를 사용하여 클러스터링 결과를 2차원 공간에 시각화하는 코드를 포함합니다. 이를 통해 클러스터 분포를 확인합니다.
    *   **사전 계산 (Pre-computation)**: 클러스터링은 Q-러닝 에이전트 학습이나 추론 이전에 **별도로 미리 수행되는 오프라인 작업**입니다. 에이전트 실행 시에는 미리 계산된 클러스터링 결과를 로드하여 활용합니다.

### 3.3. Q-러닝 기반 클러스터 선택

이 프로젝트에서는 두 가지 주요 Q-러닝 구현 방식이 확인됩니다:

**1. 단일 에이전트 (`qlearning.py` 구현)**:

*   **에이전트**: 단일 `QAgent` 클래스.
*   **상태 (State)**:
    *   각 클러스터별 최고 유사도 점수를 구간(bin)으로 변환한 값들의 **단일 튜플** (`tuple`).
    *   `binSimilarityScore` 함수: `[0, 5, 15, 25]` 구간 기준 5개 레벨 변환.
*   **행동 (Action)**: 다음에 추측할 클러스터의 인덱스 (단일 값).
*   **보상 (Reward)**: 선택된 클러스터에서 휴리스틱으로 추측한 단어의 **실제 유사도 점수** (휴리스틱 함수의 반환값).
*   **Q-테이블 (Q-Table)**: `q_table[state][action]` 형태의 딕셔너리.
*   **Q-값 업데이트**: 표준 Q-러닝 업데이트 공식을 사용 (\( \alpha=0.3, \gamma=0.75 \)).
*   **탐험/활용**: 엡실론-탐욕 및 엡실론 감쇠 사용.

**2. 계층적 에이전트 시도 (`qlearning.ipynb` 구현 부분 분석)**:

*   **에이전트**: 계층적 구조를 위한 `Environment` 클래스 변경 (별도 `QAgent` 클래스 구조는 추가 확인 필요).
*   **상태 (State)**:
    *   **상위 상태 (`state`)**: `qlearning.py`와 유사하게 상위 클러스터 상태 관리.
    *   **하위 상태 (`sub_state`)**: 각 상위 클러스터 내의 **하위 클러스터별 최고 유사도 점수를 구간으로 변환**한 값들의 **2차원 리스트** (`list[list[int]]`).
    *   `bin_similarity_score` 함수: `[0, 15, 25, 50]` 구간 기준 4개 레벨 변환.
*   **행동 (Action)**: 명시적 정의는 추가 확인 필요하나, `guess_word` 함수가 `top_action`(상위 클러스터)과 `sub_action`(하위 클러스터)을 모두 받는 것으로 보아 계층적 행동 선택 로직 포함.
*   **보상 (Reward)**: **선택한 (상위) 클러스터에 실제 정답 단어가 포함되어 있는지 여부**에 따라 큰 보상/패널티를 주는 방식 (`clusterReward`)으로 변경. 실제 유사도 점수 기반 보상은 제거됨.
*   **Q-테이블 및 업데이트**: 구체적인 Q-테이블 구조 및 업데이트 방식은 해당 노트북의 추가 분석 필요.

### 3.4. 휴리스틱 기반 단어 선택

Q-러닝 에이전트가 추측할 클러스터를 선택하면, 실제 단어는 다음 휴리스틱 함수들 중 하나를 통해 선택됩니다:

*   **탐욕적 휴리스틱 (`qlearning.py`의 `choose_word_from_cluster` 함수)**:
    *   **구현**: `qlearning.py`에 구현된 기본 휴리스틱입니다.
    *   **로직**: 선택된 클러스터 내에서 이전 추측 중 최고/최저 유사도 단어를 기준으로 가장 유망한 단어를 선택합니다. 이전 추측 최고점(`best_similarity`)이 임계값(20)보다 낮으면 최고/최저 유사도 단어와의 평균 유사도가 높은 단어를, 높으면 최고 유사도 단어와의 유사도에서 최저 유사도 단어와의 유사도를 뺀 값이 가장 큰 단어를 선택합니다. 이는 탐색(낮은 점수)과 활용(높은 점수) 간의 균형을 맞추려는 시도입니다.

*   **RNN 기반 휴리스틱 (`src/heuristicrnn.py`의 `RNN` 클래스 및 `src/heuristic.ipynb`)**:
    *   **구현**: `heuristicrnn.py`는 PyTorch(`torch.nn`)를 사용하여 RNN 모델(`class RNN`)을 정의하며, 모델 구조 정의(`__init__`), 순전파(`forward`), 손실 계산(`compute_loss`), 모델 저장/로드 기능을 포함합니다.
    *   **학습 (`heuristic.ipynb`)**: `heuristic.ipynb`는 이 RNN 모델을 학습시키는 코드입니다.
        *   **데이터 생성**: 무작위 게임 플레이 데이터를 시뮬레이션하여 입력(과거 추측 벡터+점수 시퀀스)과 출력(다음 추측 단어 벡터) 쌍을 생성하고 JSON 파일(`train.json`, `val.json`)로 저장합니다.
        *   **학습 과정**: 생성된 데이터를 사용하여 PyTorch 학습 루프를 실행합니다. 입력 시퀀스를 받아 다음 추측 단어의 벡터를 예측하도록 모델을 학습시키며, 코사인 유사도 손실 또는 MSE 손실을 사용합니다. 최적 모델은 검증 손실을 기준으로 저장됩니다.
    *   **역할**: 이전 추측 단어들과 점수 시퀀스를 기반으로 다음 추측 단어의 벡터를 예측하는 방식으로 작동합니다.
    *   **결과**: README에 따르면, 이 RNN 휴리스틱은 탐욕적 휴리스틱의 대안으로 실험되었으나 성공률과 추측 속도 면에서 낮은 성능을 보였습니다.

*   **대체 휴리스틱 (`Testing/heuristicAlternatives.py`)**:
    *   **구현**: 이 파일에는 `choose_word_from_cluster` 및 `choose_word_from_cluster_modified`와 같은 대체 휴리스틱 함수들이 포함되어 있습니다.
    *   **역할**: `qlearning.py`의 기본 탐욕적 휴리스틱 외에 다른 단어 선택 전략(예: 유사도 행렬 사용)을 구현하고 비교 실험하는 데 사용된 코드입니다.

### 3.5. 학습 과정 상세

Q-러닝 에이전트는 여러 게임 에피소드를 플레이하면서 최적의 클러스터 선택 정책을 학습합니다. 학습 과정은 다음과 같습니다:

1.  **초기화**: Q-테이블을 0으로 초기화하고, 엡실론(ε) 값을 1.0으로 설정합니다 (`QAgent.__init__`).
2.  **에피소드 반복**: 지정된 게임 수 (`num_games`)만큼 다음 과정을 반복합니다 (`qlearning.py` 메인 루프).
    a.  **게임 시작**: 새로운 비밀 단어(`mystery_word` from `answers.py`)를 설정하고 환경(`environment`)을 초기 상태로 리셋합니다.
    b.  **스텝 반복**: 최대 추측 횟수(`max_guesses`) 또는 게임 종료 시까지 다음을 반복합니다.
        i.  **현재 상태 확인**: 현재 환경의 상태 `s`를 얻습니다 (`environment.get_state`).
        ii. **행동 선택**: 엡실론-탐욕 정책에 따라 행동 `a` (클러스터 인덱스)를 선택합니다 (`QAgent.choose_action`).
        iii. **단어 선택 및 추측**: 선택된 클러스터 `a`에서 휴리스틱 함수(기본적으로 `choose_word_from_cluster`)를 사용하여 단어 `w`를 선택하고 추측합니다 (`environment.guess_word`).
        iv. **보상 및 다음 상태 관찰**: 추측 결과로 보상 `r` (유사도 점수)과 다음 상태 `s'`를 얻습니다 (`environment.guess_word` 반환값 및 `environment.get_state`).
        v.  **Q-테이블 업데이트**: 관찰된 `(s, a, r, s')` 정보를 사용하여 Q-러닝 업데이트 규칙에 따라 `Q(s, a)` 값을 업데이트합니다 (`QAgent.update`).
        vi. **상태 전이**: 현재 상태를 `s'`로 업데이트합니다.
c.  **엡실론 감쇠**: 한 게임 에피소드가 끝나면 엡실론 값을 감소시킵니다 (`QAgent.decayEpsilon`).
3.  **학습 완료**: 모든 게임 에피소드가 완료되면 Q-테이블에는 각 상태에서 어떤 클러스터를 선택하는 것이 가장 좋은지에 대한 학습된 Q-값이 저장됩니다.

### 3.6. 추론 과정 상세 (학습 후 게임 플레이)

학습이 완료된 Q-테이블을 사용하여 실제 Semantle 게임을 플레이하는 과정(추론)은 다음과 같습니다:

1.  **학습된 Q-테이블 로드/사용**: 학습 과정에서 생성된 Q-테이블을 사용합니다.
2.  **엡실론 설정**: 탐험을 최소화하고 학습된 최적 정책을 활용하기 위해 엡실론(ε) 값을 매우 작게 설정합니다 (예: 0 또는 0.01).
3.  **게임 시작**: 새로운 비밀 단어를 설정하고 환경을 초기 상태로 리셋합니다.
4.  **스텝 반복**: 게임 종료 시까지 다음을 반복합니다.
    a.  **현재 상태 확인**: 현재 환경의 상태 `s`를 얻습니다 (`environment.get_state`).
    b.  **최적 행동 선택**: 현재 상태 `s`에서 Q-테이블을 참조하여 **가장 높은 Q-값을 가지는 행동 `a` (클러스터 인덱스)**를 선택합니다 (`QAgent.choose_action`에서 `np.argmax` 사용).
    c.  **단어 선택 및 추측**: 선택된 최적 클러스터 `a`에서 휴리스틱 함수(`choose_word_from_cluster` 등)를 사용하여 단어 `w`를 선택하고 추측합니다 (`environment.guess_word`).
    d.  **상태 전이**: 추측 결과에 따라 다음 상태 `s'`로 이동합니다.
5.  **게임 종료**: 비밀 단어를 맞추거나 최대 추측 횟수에 도달하면 게임이 종료됩니다.

추론 과정에서는 더 이상 Q-테이블을 업데이트하지 않고, 오직 학습된 Q-값을 사용하여 가장 효율적인 클러스터를 선택하는 데 집중합니다.

## 4. 전체 워크플로우 요약

1.  **(사전 준비)** `answers.py`의 정답 목록을 사용하여 단어 클러스터링 수행 (`sklearn.cluster.AgglomerativeClustering` 또는 `KMeans` 등 사용) 및 결과 저장 (`clusters.py`, `secondLevelClusters.py` 등 생성. `Testing/clusterAl.py`, `src/cluster.ipynb` 참고).
2.  **(학습 단계)** `qlearning.py`를 실행하여 다수의 게임 플레이를 통해 Q-테이블 학습.
3.  **(추론/테스트 단계)** 학습된 Q-테이블과 낮은 엡실론 값을 사용하여 실제 게임 플레이 (예: `Testing/runSingle.py` 사용). 또한, `Testing/Semantle.py`를 통해 사용자가 직접 게임 플레이 가능.

## 5. 주요 결과 (README 기반)

*   탐욕적 휴리스틱을 사용한 AI는 50번의 추측 내에서 **79.2%**의 게임을 성공적으로 해결했습니다.
*   RNN 기반 휴리스틱은 성공률(50.1%)과 추측 속도 면에서 탐욕적 방식보다 성능이 낮았습니다.
*   엡실론 감쇠 전략은 효과적이어서, 10,000 게임 동안 Q-에이전트가 가능한 상태의 상당 부분(약 600/625개)을 탐색했습니다.

## 6. README와 코드 구현 간의 확인된 차이점 및 연결점

`README.md`의 설명과 실제 코드 구현 사이에는 다음과 같은 명확한 차이점 및 연결점이 존재합니다:

*   **Q-에이전트 구조**: `README.md`는 "계층적 Q-에이전트" 구조를 언급합니다. `qlearning.py`에는 **단일 `QAgent` 클래스**만 구현되어 있지만, **`src/qlearning.ipynb`에서는 `secondLevelClusters.py` 데이터를 활용하여 계층적 상태 및 관련 로직을 구현하려는 명확한 시도**가 확인됩니다. 이는 README의 설명을 뒷받침하는 실험 코드가 존재함을 의미합니다.

*   **사용된 휴리스틱 함수**: `README.md`는 탐욕적 휴리스틱과 RNN 휴리스틱을 모두 언급합니다. `qlearning.py`의 핵심 학습 루프는 **탐욕적 휴리스틱 함수만 명시적으로 호출**하지만, RNN 휴리스틱 구현(`heuristicrnn.py`)과 대체 휴리스틱 구현(`Testing/heuristicAlternatives.py`)이 별도로 존재합니다. **`qlearning.ipynb`에서는 `heuristicrnn`과 `torch`를 임포트하는 것으로 보아 RNN 휴리스틱을 통합했을 가능성**이 있습니다.

*   **클러스터링**: `README.md`에서 언급된 **응집형 계층적 클러스터링**은 `Testing/clusterAl.py`에서 `sklearn` 라이브러리를 통해 구현된 것을 확인할 수 있으며, `src/cluster.ipynb`에서는 **KMeans 클러스터링** 사용이 확인됩니다. 이는 프로젝트에서 여러 클러스터링 방법을 시도했음을 보여줍니다. `qlearning.py`는 단일 레벨 클러스터(`clusters.py`)를, **`qlearning.ipynb`는 계층적 클러스터(`secondLevelClusters.py`)를 사용**하는 것으로 보입니다.

이러한 차이점들은 `README.md`가 프로젝트의 전체적인 설계와 다양한 실험(계층적 에이전트, RNN 휴리스틱, 다양한 클러스터링 방법 등)을 포괄적으로 설명하는 반면, `qlearning.py`는 특정 버전(탐욕적 휴리스틱을 사용한 단일 Q-에이전트 학습)의 코드를 보여주는 것으로 해석됩니다.

## 7. 기타 관련 파일 분석 (확인된 내용)

*   **`src/answers.py`**: Semantle 게임의 가능한 정답 단어 목록인 `secretWords` 변수를 파이썬 리스트 형태로 정의하고 있습니다. `qlearning.py` 및 테스트 코드들에서 게임 환경 설정 시 이 리스트를 사용합니다.
*   **`Testing/heuristicAlternatives.py`**: `qlearning.py`의 기본 탐욕적 휴리스틱 외에, `choose_word_from_cluster_modified` 등 다른 단어 선택 전략을 구현한 함수들을 포함하고 있습니다. 이는 휴리스틱 전략 비교 실험에 사용된 코드입니다.
*   **`Testing/runSingle.py`**: `run_single_game` 함수를 정의하고 있습니다. 이 함수는 학습된 Q-에이전트(`q_agent`)와 특정 설정을 이용하여 단일 Semantle 게임을 실행하고 결과를 관찰/평가하는 데 사용됩니다.
*   **`Testing/clusterAl.py`**: `sklearn`의 `AgglomerativeClustering`을 사용하여 `answers.py`의 단어 목록에 대한 클러스터링을 수행하는 예시 코드를 포함합니다. 클러스터링 파라미터(8개 클러스터, 유클리드 거리, ward 연결)를 확인할 수 있습니다.
*   **`Testing/Semantle.py`**: Tkinter GUI를 사용하여 사용자가 직접 Semantle 게임을 플레이할 수 있는 환경을 제공합니다. AI 솔버와는 별개인 게임 자체의 구현입니다.
*   **`src/cluster.ipynb`**: `sklearn`의 `KMeans` 클러스터링 알고리즘을 사용하여 `answers.py`의 단어 목록을 클러스터링하고, UMAP과 Plotly를 이용하여 결과를 시각화하는 Jupyter Notebook입니다.
*   **`src/qlearning.ipynb`**: **계층적 Q-러닝 구조를 실험한 것으로 보이는 Jupyter Notebook**입니다. `secondLevelClusters.py` 데이터를 사용하고, 계층적 상태(`sub_state`) 및 다른 보상 함수(`clusterReward`)를 정의하는 등 `qlearning.py`와는 다른 구현 방식을 포함합니다.
*   **`src/parallelQ.ipynb`**: `qlearning.py`의 **단일 에이전트 Q-러닝 학습 과정을 병렬화**하여 속도를 높이려는 시도를 보여주는 Jupyter Notebook입니다. `concurrent.futures`를 사용하여 여러 게임 에피소드를 동시에 실행하고 공유 Q-테이블을 업데이트합니다. 계층적 구조는 사용하지 않으며, `qlearning.py`와는 다른 보상 함수(`get_reward`)를 사용합니다.
*   **`src/heuristic.ipynb`**: **RNN 기반 휴리스틱 모델(`heuristicrnn.py`)을 학습시키는 Jupyter Notebook**입니다. 학습/검증 데이터 생성(`generate_data`), 데이터 전처리(`get_data_from_json`, `SemantleDataset`), PyTorch 기반 RNN 모델 학습 및 평가 코드를 포함합니다.

이 분석은 AI-Semantle 프로젝트가 Q-러닝, 클러스터링, 휴리스틱을 효과적으로 결합하여 복잡한 자연어 기반 게임인 Semantle을 해결하는 방식을 보여줍니다. 또한 단일 에이전트와 계층적 에이전트, 다양한 휴리스틱과 클러스터링 방법을 실험하며 최적의 접근법을 탐색했음을 시사합니다.

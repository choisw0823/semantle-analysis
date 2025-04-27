# Pedantle-RL 프로젝트 분석

제공해주신 파일들을 바탕으로 Pedantle-RL 프로젝트를 분석한 결과는 다음과 같습니다.

## 1. 프로젝트 목적

*   이 프로젝트는 **Pedantle**이라는 웹 게임을 플레이하는 강화학습(RL) 에이전트를 개발하는 것을 목표로 합니다.
*   Pedantle 게임은 매일 무작위로 선정된 Wikipedia 기사의 제목을 맞추는 게임입니다. 플레이어는 단어를 제출하고, 제출한 단어가 기사 본문에 있는 단어들과 얼마나 유사한지에 대한 피드백을 받습니다.
*   프로젝트는 Google의 **Word2Vec** 모델을 사용하여 단어 간의 의미론적 유사성을 계산하고, 이를 기반으로 에이전트가 최적의 단어를 추측하도록 학습시킵니다.

## 2. 주요 구성 요소 및 기술 스택

### 강화학습 환경 (Gym Environment)

*   **위치:** `gym-examples/` 디렉토리
*   **구현:** OpenAI Gym 프레임워크 기반 커스텀 Pedantle 게임 환경 (`PedantleEnv` in `envs/pedantle.py`)
*   **상태 (Observation):**
    *   기사 제목 및 본문 단어들의 발견 여부
    *   가장 가까운 제안 단어와의 유사도 (proximity)
    *   단어 길이
    *   현재까지 제안된 단어 목록
    *   기사/제목에 포함된 것으로 추정되는 단어 목록
    *   복합적인 상태 공간 (`spaces.Dict`)
*   **행동 (Action):**
    *   에이전트가 다음에 추측할 단어를 문자열 형태로 선택 (`spaces.Text`)
*   **보상 (Reward):**
    *   제안 단어가 실제 제목/본문 단어와 매우 유사 (정답 발견) 또는 어느 정도 유사할 경우 **양의 보상**
    *   그렇지 않거나 이미 제안한 단어일 경우 **음의 보상**
    *   제목 전체를 맞추면 **큰 보상**
*   **렌더링:**
    *   Pygame을 사용하여 게임 진행 상황 시각화 (`render_mode="human"`)

### 강화학습 에이전트 (RL Agent)

*   **위치:** `agent/` 디렉토리
*   **알고리즘:** Q-러닝 기반 (`QLearningAgent` in `agent/Q_learning_agent.py`)
*   **상태 표현:**
    *   환경에서 받은 복잡한 관측(observation)을 이산적인 상태 값(0~99)으로 변환 (`agent/states.py`의 `compute_state` 함수)
    *   Q-테이블 인덱스로 사용 (주로 발견된 단어 비율 기반)
*   **행동 선택:**
    *   ε-greedy 정책 사용
    *   `agent/actions.py`에 정의된 다양한 행동 전략 중 선택:
        *   고전적인 단어 목록 사용
        *   첫 번째 미발견 단어 추측
        *   무작위 단어 추측
        *   마지막 타겟 단어 기반 추측
        *   제목 단어 기반 추측 등
    *   실제 제안 단어 생성: `agent/propose_words.py`
*   **학습:**
    *   환경과의 상호작용 및 보상을 통해 Q-테이블 업데이트
    *   최적의 행동 정책 학습

### 단어 임베딩 및 유사도 계산

*   **라이브러리:** `gensim`
*   **모델:** 사전 학습된 Google News Word2Vec (`data/GoogleNews-vectors-negative300.bin`)
*   **위치:** `gym_examples/wrappers/sim_computer.py`
*   **기능:**
    *   Word2Vec 모델 로드
    *   두 단어 벡터 간 코사인 유사도 계산 (`compute_similarity` 함수)

### 유사 단어 검색 (Faiss)

*   **라이브러리:** `faiss`
*   **인덱스 생성:** `faiss_index_maker.py`
    *   Word2Vec 벡터에 대한 효율적인 검색 인덱스 생성
    *   HNSW 알고리즘과 내적(Inner Product) 메트릭 사용
*   **인덱스 종류:**
    *   테스트용 소규모 인덱스: `data/word2vec_test.faiss`
    *   전체 모델용 대규모 인덱스: `data/v3_cpu_word2vec_full.faiss`
*   **활용:**
    *   특정 단어와 가장 유사한 단어들을 빠르게 검색 (`agent/utils.py`의 `get_nearest_words`)

### 데이터

*   `data/wikipedia_april.csv`: 기본적인 테스트를 위한 샘플 Wikipedia 기사 데이터.
*   `dataset_downloader.py`: `wikipedia` 라이브러리를 사용하여 특정 주제의 Wikipedia 기사를 다운로드하고 `data/wikipedia_dataset.csv`로 저장하는 스크립트.
*   `data/GoogleNews-vectors-negative300.bin`: Google의 사전 학습된 Word2Vec 모델 파일 (별도 다운로드 필요).

### 실행 및 결과 분석

*   `main.py`: 에이전트 학습 및 실행을 위한 메인 스크립트.
    *   환경과 에이전트 초기화
    *   여러 에피소드에 걸쳐 학습 진행
    *   학습 과정 로그 기록 (`logs/`)
*   `results.py`: 학습 완료 후 저장된 결과(`results.json`) 시각화.
    *   `matplotlib` 사용
    *   Q-값, 에피소드별 소요 단어 수, 상태 방문 횟수 등 시각화

## 3. 종합 의견

*   Pedantle-RL은 **자연어 처리**와 **강화학습**을 결합하여 복잡한 게임 환경을 해결하려는 흥미로운 시도입니다.
*   **Word2Vec**과 **Faiss**를 효과적으로 활용하여 단어 의미 유사성 기반의 게임 플레이 및 효율적인 검색을 구현했습니다.
*   **Gym 프레임워크**를 사용하여 표준화된 방식으로 환경을 구축했으며, **Q-러닝 에이전트**와 다양한 행동 전략을 구현하여 문제에 접근하고 있습니다.
*   **상태 공간 정의**와 **행동 전략 설계**가 에이전트 성능에 중요한 영향을 미칠 것으로 보입니다.
*   코드 구조가 비교적 잘 분리되어 있어 (환경, 에이전트, 유틸리티 등) **이해하고 수정하기 용이**합니다.

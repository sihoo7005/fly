# FlyWire 뇌 모델

FlyWire v783 연결망으로 새 모델을 만들기 위한 시작 저장소입니다.
데이터 다운로드와 직접 작성한 발화 시뮬레이터가 있습니다.
외부 패키지를 사용하지 않으며, Python 표준 라이브러리만 사용합니다.

## 데이터 받기

Python 3.9 이상과 약 1.2 GB의 저장 공간이 필요합니다. 추가 패키지는 필요 없습니다.

네 원본 파일은 [FlyWire v783 데이터 릴리스](https://github.com/sihoo7005/fly/releases/tag/flywire-v783)의 Assets에서도 받을 수 있습니다.
큰 파일은 Releases에 보관하며, **Code → Download ZIP**에는 포함되지 않습니다.

Windows PowerShell에서:

```powershell
git clone https://github.com/sihoo7005/fly.git
cd fly
py download_data.py
```

Git이 없다면 GitHub의 **Code → Download ZIP**으로 받고 압축을 푼 뒤,
그 폴더에서 PowerShell을 열고 `py download_data.py`를 실행합니다.
Python이 `python` 명령으로 설치된 환경에서는 `python download_data.py`를 사용합니다.
Linux와 Termux에서는 `python3 download_data.py`를 사용할 수 있습니다.

파일은 스크립트 옆의 `data/raw/` 폴더에 저장됩니다.

| 파일 | 용량 | 내용 |
|---|---:|---|
| `proofread_connections_783.feather` | 약 852 MB | 교정된 뉴런 사이의 연결과 영역별 시냅스 수 |
| `proofread_root_ids_783.npy` | 약 1.11 MB | 교정된 뉴런 식별번호 목록 |
| `per_neuron_neuropil_count_pre_783.feather` | 약 16.85 MB | 뉴런별·뇌 영역별 출력 시냅스 수 |
| `per_neuron_neuropil_count_post_783.feather` | 약 233.84 MB | 뉴런별·뇌 영역별 입력 시냅스 수 |

완료 시 배포본의 파일 크기와 MD5 체크섬을 검증합니다.
이미 검증된 파일은 다시 받지 않습니다. 중단된 파일은 다음 실행에서 처음부터 받습니다.
`data/`는 Git 업로드 대상에서 제외됩니다.

## 데이터 구조

연결표의 한 행은 출발 뉴런, 도착 뉴런, 뇌 영역의 조합입니다.
`pre_pt_root_id`, `post_pt_root_id`, `neuropil`, `syn_count`와
전달물질별 평균 예측 확률이 들어 있습니다.
같은 뉴런 쌍이 여러 영역에 나타나면 `syn_count`를 합산합니다.
이 표는 시냅스가 1개 이상인 연결을 포함합니다.
식별번호는 64비트 정수로 보관해야 합니다.

이 데이터에는 실제 시냅스 강도나 뉴런 발화 규칙이 없습니다.
아래 모델의 가중치와 동작 규칙은 별도로 정한 가정입니다.
영역별 입력·출력 집계 파일은 미교정 세그먼트도 포함한 전체 시냅스 표에서 집계한 자료입니다.
교정된 뉴런만 분석할 때는 `proofread_root_ids_783.npy`로 필터링해야 합니다.

## 모델 실행

먼저 작은 예제로 신호 전달을 확인합니다. 데이터 다운로드 없이 실행됩니다.

```shell
python model.py demo
```

네 뉴런이 `1 → 2 → 3 → 4` 순서로 발화합니다.
출력은 `results/spikes.csv`에 시간(ms)과 뉴런 식별번호로 저장됩니다.

실제 FlyWire 데이터는 **Python 3.14 이상과 표준 모듈 `compression.zstd`**가
필요합니다. Feather 구조는 직접 해석하고, Zstandard 압축만 이 표준 모듈로 풉니다.
외부 Arrow, NumPy, pandas, 뇌 시뮬레이터 패키지는 사용하지 않습니다.
배포판에 해당 표준 모듈이 없는 경우 실행 중 필요한 기능을 알려주며,
다른 라이브러리를 자동 설치하거나 사용하지 않습니다.

Python 3.14 이상을 선택한 환경에서:

```shell
python download_data.py
python model.py build
python model.py run
```

Windows에서 여러 Python 버전이 있다면 각 명령의 `python`을 `py -3.14`로 바꿉니다.
`build`는 뉴런 목록과 연결표를 두 번에 걸쳐 스트리밍으로 읽고,
영역별로 반복된 뉴런 쌍을 합산하여 `data/network.bin`에 저장합니다.
연결이 있는 항목만 배열로 저장하며, 모든 뉴런 쌍의 거대한 행렬을 만들지 않습니다.
이후 `run`은 이 캐시를 사용하므로 다시 원본을 읽을 필요가 없습니다.
입력·출력 집계표는 이 모델의 연결 생성에 필요하지 않은 보조자료입니다.
`flywire_data.py`의 `FeatherReader`로 집계표도 읽을 수 있습니다.

입력 뉴런을 지정하려면:

```shell
python model.py run --input 720575940621014292 --duration 100 --pulse-every 10 --weight 0.01
```

입력을 지정하지 않으면 서로 다른 출력 연결이 가장 많은 뉴런 하나를 선택하고
그 식별번호를 표시합니다. 이는 실행 예제를 위한 선택이며 감각 뉴런을 식별한 결과가 아닙니다.
기본적으로 0, 10ms마다 입력 뉴런에 임계값만큼의 자극을 넣습니다.
파일에는 **발화 기록**을 저장하며, 매 시간 단계의 모든 활동값을 저장하지는 않습니다.

## 모델 규칙과 가정

- 같은 뉴런 쌍의 시냅스 수를 합산하고, 가중치 크기를 `k × syn_count`로 정합니다.
- 활동값은 마지막 갱신 이후 시간에 따라 `exp(-시간 / tau)`만큼 감소합니다.
- 외부 입력과 도착한 연결 신호를 더해 임계값 이상이면 발화합니다.
- 발화하면 활동값을 0으로 초기화하고 지정한 시간 동안 입력을 무시합니다.
- 발화 신호는 **한 시간 단계 뒤**에 전달됩니다. 모든 뉴런을 같은 시점 기준으로 갱신합니다.
- 시간 단계·누수 시간·초기화 규칙·임계값·가중치 상수는 생물학적으로 보정하지 않은 가정입니다.
- **기본 실행은 모든 뉴런을 흥분성으로 가정합니다.** 전달물질 예측값으로 부호를 자동 분류하지 않습니다.
  `--inhibitory` 뒤에 식별번호를 나열하면 해당 뉴런의 출력 가중치에 음수 부호를 붙입니다.

| 설정 | 기본값 | 명령 옵션 |
|---|---:|---|
| 시냅스 하나당 가중치 크기 | 0.01 | `--weight` |
| 시간 단계 | 1ms | `--dt` |
| 누수 시간 상수 | 20ms | `--tau` |
| 발화 임계값 | 1 | `--threshold` |
| 발화 후 입력을 무시하는 시간 | 2ms | `--refractory` |
| 실행 시간 | 20ms | `--duration` |
| 자극 간격 | 10ms | `--pulse-every` |

활동값과 가중치는 무차원 값입니다. 실제 막전위나 전류로 해석하지 않습니다.
자극 간격은 시간 단계의 정수배여야 합니다. 발화 후 대기 시간은 시간 단계 단위로 올림합니다.

## 출처

- [FlyWire v783 공식 배포본과 라이선스](https://zenodo.org/records/10676866)
- [FlyWire 인용 안내](https://flywire.ai/guidelines)

## 코드 검사

```shell
python -m unittest discover -v
```

다운로드 검증, 정확한 64비트 식별번호 읽기, 압축 해제,
영역별 시냅스 합산, 가중치 비례, 신호 전달 지연, 누수, 억제, 발화 후 대기,
캐시 저장과 손상 파일 거부를 검사합니다.
Python 3.14 표준 압축 기능이 없으면 해당 압축 검사만 건너뜁니다.
GitHub Actions는 캐시된 Python 3.14로 모든 검사를 실행하고,
실제 FlyWire 뉴런 목록과 연결표의 체크섬을 검증한 뒤 전체 모델을 생성·실행합니다.

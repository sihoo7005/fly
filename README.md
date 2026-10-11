# FlyWire 뇌 모델

FlyWire v783 연결망으로 새 모델을 만들기 위한 시작 저장소입니다.
현재는 데이터 다운로드 기능만 있습니다. 뇌 시뮬레이터는 아직 없습니다.

## 데이터 받기

Python 3.9 이상과 약 1 GB의 저장 공간이 필요합니다. 추가 패키지는 필요 없습니다.

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
가중치와 동역학은 모델 구현 단계에서 별도로 정합니다.

## 출처

- [FlyWire v783 공식 배포본과 라이선스](https://zenodo.org/records/10676866)
- [FlyWire 인용 안내](https://flywire.ai/guidelines)

## 다운로드 코드 검사

```shell
python -m unittest test_download_data.py
```

작은 가짜 응답으로 정상 다운로드, 재사용, 손상 파일 거부를 검사합니다.

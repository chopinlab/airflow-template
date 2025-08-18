# Airflow MLOps Template

Apache Airflow와 MLflow를 기반으로 한 완전한 MLOps 파이프라인 템플릿입니다.

## 📁 현재 프로젝트 구조

```
airflow-template/
├── dags/                    # Airflow DAG 파일들
├── mlflow-projects/         # MLflow 프로젝트 정의
├── data/                    # 훈련/테스트 데이터
├── artifacts/               # MLflow 아티팩트
├── models/                  # 저장된 모델
├── logs/                    # 실행 로그
├── plugins/                 # 커스텀 플러그인
├── scripts/                 # 유틸리티 스크립트
└── docker-compose.yml       # 서비스 오케스트레이션
```

## 🏗️ 아키텍처 개선 방안

### 현재 구조의 한계점

1. **관심사 분리 부족**: Airflow와 ML 코드가 혼재
2. **확장성 제한**: 새로운 ML 프로젝트 추가 시 복잡성 증가
3. **유지보수성**: 오케스트레이션과 비즈니스 로직이 결합

### 권장 아키텍처 옵션

#### Option A: 마이크로서비스 분리 (Enterprise 환경)
```
airflow-template/               # 오케스트레이션 전용
├── dags/
└── docker-compose.yml

ml-training-service/            # 별도 리포지토리
├── src/
├── models/
└── api/

ml-serving-service/             # 별도 리포지토리
├── src/
└── api/
```

#### Option B: 모노레포 구조 개선
```
airflow-template/
├── services/
│   ├── training/              # 훈련 서비스
│   ├── serving/               # 서빙 서비스
│   └── data-processing/       # 데이터 처리
├── dags/                      # 오케스트레이션만
└── shared/                    # 공통 라이브러리
```

#### Option C: 현재 구조 점진적 개선
```
airflow-template/
├── dags/                      # Airflow DAGs (오케스트레이션만)
├── ml-services/              # mlflow-projects 대신
│   └── image-classification/
└── shared-libs/              # 공통 유틸리티
```

### 개선 원칙

1. **단일 책임 원칙**: DAG는 워크플로우 오케스트레이션만 담당
2. **느슨한 결합**: ML 서비스는 API를 통해서만 통신
3. **확장 가능성**: 새로운 ML 프로젝트 추가 시 독립적 개발 가능
4. **재사용성**: 공통 컴포넌트는 shared 모듈로 분리

### 마이그레이션 전략

1. **1단계**: `mlflow-projects` → `ml-services` 리네임
2. **2단계**: 공통 유틸리티를 `shared-libs`로 분리
3. **3단계**: 각 ML 서비스를 독립적인 컨테이너로 분리
4. **4단계**: 필요시 별도 리포지토리로 완전 분리

## 🚀 빠른 시작

### 필수 요구사항
- Docker 및 Docker Compose 설치
- Python 3.12+ (로컬 개발 시)
- Apache Airflow 기본 지식

### 설치 및 실행

1. **필수 디렉토리 생성**:
   ```bash
   mkdir -p dags logs plugins config
   ```

2. **환경 변수 설정** (Linux/WSL):
   ```bash
   echo -e "AIRFLOW_UID=$(id -u)" > .env
   ```

3. **서비스 시작**:
   ```bash
   docker-compose up
   ```

4. **서비스 접속**:
   - Airflow Web UI: http://localhost:8080 (admin/admin)
   - MLflow Web UI: http://localhost:5000
   - Model Serving API: http://localhost:5001

### 주요 명령어

```bash
# 서비스 시작 (백그라운드)
docker-compose up -d

# 서비스 중지
docker-compose down

# 로그 확인
docker-compose logs -f

# Airflow 명령 실행
docker-compose exec airflow airflow --help

# MLflow 로그 확인
docker-compose logs -f mlflow
```

## 🔄 MLOps 워크플로우

### 1. MLflow 오케스트레이션 파이프라인
- **스케줄**: 6시간마다 자동 실행
- **단계**:
  1. 샘플 데이터 생성
  2. MLflow 훈련 트리거
  3. 최신 모델 정보 조회
  4. 추론 실행
  5. 파이프라인 검증

### 2. 모델 관리 파이프라인
- **기능**:
  - 모델 목록 조회
  - Staging 모델 검증 (정확도 > 0.8)
  - Production 자동 승격
  - 오래된 버전 정리 (30일 기준)
  - 성능 모니터링 및 알림

## 🛠️ 기술 스택

- **오케스트레이션**: Apache Airflow 3.0.4
- **실험 추적**: MLflow 2.8.0+
- **ML 프레임워크**: PyTorch, scikit-learn
- **데이터베이스**: PostgreSQL 13
- **컨테이너**: Docker & Docker Compose
- **모델 서빙**: FastAPI, BentoML
- **모니터링**: Prometheus, Weights & Biases

## 📋 디렉토리 설명

- `dags/` - Airflow DAG 워크플로우 정의
- `mlflow-projects/` - MLflow 프로젝트 및 ML 코드
- `data/` - 원시 및 가공 데이터셋
- `artifacts/` - MLflow 아티팩트 및 출력물
- `models/` - 훈련된 ML 모델
- `logs/` - Airflow 실행 로그
- `plugins/` - 사용자 정의 Airflow 플러그인
- `scripts/` - 재사용 가능한 Python 스크립트
- `sql/` - SQL 쿼리 파일

## 🔍 MLflow 통합

### 서비스 구성
- **MLflow Server**: http://localhost:5000 - 실험 추적 및 모델 레지스트리
- **MLflow Training**: 백그라운드 훈련 작업 컨테이너
- **MLflow Serving**: http://localhost:5001 - FastAPI 모델 서빙 서비스
- **Backend Store**: PostgreSQL 데이터베이스
- **Artifact Store**: 로컬 파일시스템 (`./artifacts`)

### 사용 예시
```python
import mlflow
mlflow.set_tracking_uri("http://mlflow:5000")

with mlflow.start_run():
    mlflow.log_param("param1", value)
    mlflow.log_metric("metric1", value)
    mlflow.sklearn.log_model(model, "model")
```

## 📝 샘플 DAG

- `hello_world_dag.py` - 기본 Airflow 기능
- `mlflow_orchestration_pipeline.py` - MLflow 기반 ML 파이프라인 오케스트레이션
- `simple_image_classifier.py` - 간단한 이미지 분류 예제
- `model_management_dag.py` - 모델 라이프사이클 관리

## 🧪 MLflow 프로젝트

`mlflow-projects/image-classification/` - 완전한 이미지 분류 프로젝트:
- `train.py` - 모델 훈련 스크립트
- `inference.py` - 모델 추론 스크립트
- `serve.py` - FastAPI 모델 서빙 서비스
- `MLproject` - MLflow 프로젝트 설정
- `conda.yaml` - 환경 의존성

## 🚀 다음 단계

1. **구조 개선**: 위에서 제안한 아키텍처 옵션 중 하나 선택
2. **CI/CD 파이프라인**: GitHub Actions 또는 GitLab CI 통합
3. **모니터링 강화**: Prometheus + Grafana 대시보드 추가
4. **보안 강화**: RBAC, 시크릿 관리, SSL/TLS 적용
5. **확장성**: Kubernetes 배포 옵션 추가

## 📞 지원

- 이슈 리포팅: GitHub Issues
- 문서: 각 디렉토리의 README.md 참조
- 예제: `notebooks/` 디렉토리의 Jupyter 노트북

---

**참고**: 이 템플릿은 학습 및 개발 목적으로 설계되었습니다. 프로덕션 환경에서 사용하기 전에 보안, 성능, 확장성을 고려하여 적절히 수정하시기 바랍니다.
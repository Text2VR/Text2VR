# Text2VR React Frontend

React 기반의 Text2VR 웹 인터페이스입니다. 사용자가 자연어로 파노라마 장면을 설명하면 VR로 체험할 수 있는 360도 파노라마를 생성합니다.

## 주요 기능

### 🎯 핵심 기능
- **자연어 입력**: 간단한 설명으로 파노라마 장면 생성
- **실시간 상태 업데이트**: 생성 과정을 실시간으로 모니터링
- **VR 뷰어**: A-Frame을 사용한 몰입형 360도 파노라마 체험
- **반응형 디자인**: 모바일과 데스크톱에서 모두 최적화

### 🌟 VR 기능
- 마우스/터치로 둘러보기
- VR 헤드셋 지원
- 고품질 360도 파노라마 렌더링
- 2D 미리보기도 함께 제공

## 기술 스택

- **Frontend**: React 19 + TypeScript
- **3D/VR**: A-Frame
- **Build Tool**: Vite
- **Styling**: Inline CSS with Gradient Effects

## 개발 및 실행

### 개발 모드
```bash
npm run dev
```

### 빌드
```bash
npm run build
```

### 프로덕션 서버
Python FastAPI 서버가 자동으로 빌드된 React 앱을 제공합니다:
```bash
python main.py
```

## 프로젝트 구조

```
src/
├── components/
│   ├── InputForm.tsx          # 자연어 입력 폼
│   ├── StatusSection.tsx      # 생성 상태 표시
│   ├── LoadingSpinner.tsx     # 로딩 스피너
│   └── VRPanoramaViewer.tsx   # VR 파노라마 뷰어
├── services/
│   └── apiService.ts          # API 통신 서비스
├── types/
│   ├── api.ts                 # API 타입 정의
│   └── aframe.d.ts            # A-Frame 타입 정의
├── App.tsx                    # 메인 앱 컴포넌트
├── App.css                    # 글로벌 스타일
└── main.tsx                   # React 엔트리 포인트
```

## API 연동

FastAPI 백엔드와 다음 엔드포인트로 통신합니다:

- `POST /generate` - 파노라마 생성 시작
- `GET /status/{task_id}` - 생성 상태 조회
- `GET /panorama/{task_id}` - 완성된 파노라마 이미지
- `GET /health` - 서버 상태 확인

## 사용 방법

1. **장면 설명 입력**: "평화로운 숲속에서 햇빛이 나무 사이로 스며드는 모습" 같은 자연어 설명 입력
2. **장면 이름 입력** (선택): 커스텀 이름 또는 자동 생성
3. **생성 시작**: "Generate Panorama" 버튼 클릭
4. **진행 상태 확인**: 실시간으로 생성 과정 모니터링
5. **VR 체험**: 완성된 파노라마를 VR로 체험

## VR 조작법

- **마우스**: 클릭 드래그로 시점 회전
- **터치**: 터치 드래그로 시점 회전  
- **VR 헤드셋**: VR 버튼 클릭하여 몰입형 체험

## 개발 참고사항

- React 19의 최신 기능 활용
- TypeScript로 타입 안정성 확보
- A-Frame을 동적 HTML로 생성하여 TypeScript 호환성 해결
- Vite의 프록시 설정으로 개발시 API 연동
- 반응형 CSS로 모든 디바이스 대응
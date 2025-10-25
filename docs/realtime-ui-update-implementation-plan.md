# 실시간 UI 업데이트 구현 계획서
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
**작성일**: 2025-10-21
**목적**: LangGraph 파이프라인 중간 결과물을 실시간으로 프론트엔드 UI에 표시

---

## 📋 요구사항 명확화

### 핵심 요구사항

> LangGraph 파이프라인이 도는 중간중간 생성되는 3가지 결과물을 **각각 생성되는 즉시** 실시간으로 UI에 표시

#### 3가지 결과물

1. **파노라마** (`panorama.png`)
2. **세그먼테이션 결과물** (마스크, 시각화 이미지)
3. **인페인팅된 파노라마** (`inpainted_panorama.png`)

#### 현재 문제

- ❌ 전체 파이프라인이 끝나야만 UI 업데이트
- ❌ 세그먼테이션, 인페인팅 결과가 UI에 안나오거나 이상하게 표시됨

#### 목표 상태

- ✅ 파노라마 생성 완료 → 즉시 UI 표시
- ✅ 세그먼테이션 완료 → 즉시 UI 표시 (마스크/시각화)
- ✅ 인페인팅 완료 → 즉시 UI 표시

---

## 🔍 문제 원인 분석

### 근본 원인 3가지

#### 1. LangGraph 파이프라인이 동기(Synchronous) 실행됨

**위치**: `app/services/panorama_service.py:159`

```python
result = workflow.invoke(initial_state)  # 동기 실행 - 모든 노드가 끝날때까지 blocking
```

- **문제**: LangGraph의 `workflow.invoke()`는 **동기 함수**로, 모든 노드(query_rewrite → panorama_generation → segmentation → inpainting → ply_generation)가 **순차적으로 완료될 때까지 블로킹**
- **결과**: 중간 단계에서 파노라마, 세그먼테이션, 인페인팅이 생성되더라도 **전체 파이프라인이 끝나기 전까지는 함수가 리턴되지 않아** UI 업데이트가 불가능

#### 2. 중간 결과물이 task_manager에 업데이트되지 않음

**위치**: `app/workflows/nodes.py`

- **파노라마 생성 노드** (62-207행): 파노라마 생성 완료 후 `state`에만 저장, `task_manager` 업데이트 안함
- **세그먼테이션 노드** (209-295행): 세그먼테이션 완료 후 `state`에만 저장, `task_manager` 업데이트 안함
- **인페인팅 노드** (298-379행): 인페인팅 완료 후 `state`에만 저장, `task_manager` 업데이트 안함

**현재 로직**:
```python
# panorama_generation_node 예시
return {
    **state,
    "panorama_path": final_path,  # state에만 저장
    # task_manager.update_task_status(...) 호출 없음!
}
```

**검증 결과**:
```bash
$ grep "task_manager" Text2VR/app/workflows/nodes.py
# 결과: No matches found
```

**결과**: 프론트엔드가 `/status/{task_id}` API를 호출해도 `task_manager`에 중간 결과물 경로가 없어서 UI에 표시할 수 없음

#### 3. 백엔드 API에 중간 결과물 엔드포인트가 없음

**현재 상태**:
- `app/api/panorama.py` - 전체 파노라마만 반환하는 `/panorama/{task_id}` 엔드포인트만 존재
- 세그먼테이션 결과물 엔드포인트 없음
- 인페인팅된 파노라마 엔드포인트 없음

**프론트엔드 상태 인터페이스**:
```typescript
// src/types/api.ts:11-16
export interface StatusResponse {
  status: 'queued' | 'processing' | 'completed' | 'failed';
  message: string;
  task_id: string;
  progress?: number;
  // panorama_path, segmentation_data, inpainted_panorama_path 필드가 없음!
}
```

**데이터 모델**:
```python
# app/models/panorama.py:48-58
class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    message: str
    panorama_path: Optional[str] = None          # ✅ 있음
    scene_name: Optional[str] = None
    # segmentation_data_path: 필드 없음         # ❌ 없음
    # inpainted_panorama_path: 필드 없음        # ❌ 없음
```

**결과**: 프론트엔드가 중간 결과물을 받을 수 없는 구조

---

## 🏗️ 해결 방안 아키텍처

### 현재 데이터 흐름

```
Frontend (polling /status)
    ↓ (2초마다)
    └─→ Backend: GET /status/{task_id}
            ↓
            └─→ task_manager.get_task()
                    ↓
                    └─→ TaskInfo 반환 (panorama_path만 있고 중간 결과물 없음)

LangGraph Pipeline (blocking):
query_rewrite → panorama_gen → segmentation → inpainting → ply_gen
                     ↓              ↓               ↓
                  (state에만      (state에만     (state에만
                   저장됨)        저장됨)         저장됨)
                                                   ↓
                                            최종 완료 후에만
                                            task_manager 업데이트
```

### 개선된 데이터 흐름

```
Frontend (polling /status)
    ↓ (2초마다)
    └─→ Backend: GET /status/{task_id}
            ↓
            └─→ task_manager.get_task()
                    ↓
                    └─→ StatusResponse {
                          panorama_path: "...",           ✅
                          segmentation_path: "...",       ✅
                          inpainted_panorama_path: "..." ✅
                        }

LangGraph Pipeline:
query_rewrite → panorama_gen → segmentation → inpainting → ply_gen
                     ↓              ↓               ↓
                  state +        state +        state +
                  task_manager   task_manager   task_manager
                  ✅ 실시간      ✅ 실시간      ✅ 실시간
                     업데이트       업데이트       업데이트
```

---

## 📝 세부 Task 정의

### Phase 1: 백엔드 데이터 모델 확장 (필수 기반 작업)

#### Task 1.1: Pydantic 모델 확장

**파일**: `app/models/panorama.py`

**작업 내용**:
- `TaskInfo` 모델에 필드 추가:
  - `segmentation_results_path: Optional[str] = None` - JSON 경로
  - `segmentation_visualization_path: Optional[str] = None` - 시각화 이미지
  - `inpainted_panorama_path: Optional[str] = None`
- `StatusResponse` 모델에 동일 필드 추가

**변경 예시**:
```python
class TaskInfo(BaseModel):
    """Internal task information"""
    task_id: str
    status: TaskStatus
    message: str
    panorama_path: Optional[str] = None
    segmentation_results_path: Optional[str] = None              # 추가
    segmentation_visualization_path: Optional[str] = None        # 추가
    inpainted_panorama_path: Optional[str] = None                # 추가
    scene_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    request: PanoramaRequest
    error_details: Optional[str] = None

class StatusResponse(BaseModel):
    """Response model for task status"""
    task_id: str
    status: TaskStatus
    message: str
    panorama_path: Optional[str] = None
    segmentation_results_path: Optional[str] = None              # 추가
    segmentation_visualization_path: Optional[str] = None        # 추가
    inpainted_panorama_path: Optional[str] = None                # 추가
    scene_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
```

**예상 소요**: 15분
**의존성**: 없음
**검증**: Pydantic validation 통과 확인

---

#### Task 1.2: TaskManager 업데이트 로직 수정

**파일**: `app/services/task_manager.py`

**작업 내용**:
- `update_task_status()` 메서드에 파라미터 추가:
  - `segmentation_results_path: Optional[str] = None`
  - `segmentation_visualization_path: Optional[str] = None`
  - `inpainted_panorama_path: Optional[str] = None`
- 각 필드 업데이트 로직 구현
- `updated_at` 타임스탬프 갱신

**변경 예시**:
```python
def update_task_status(
    self,
    task_id: str,
    status: TaskStatus,
    message: str,
    panorama_path: Optional[str] = None,
    segmentation_results_path: Optional[str] = None,           # 추가
    segmentation_visualization_path: Optional[str] = None,     # 추가
    inpainted_panorama_path: Optional[str] = None,             # 추가
    error_details: Optional[str] = None
) -> bool:
    """Update task status"""
    if task_id not in self._tasks:
        return False

    task = self._tasks[task_id]
    task.status = status
    task.message = message
    task.updated_at = datetime.now()

    if panorama_path:
        task.panorama_path = panorama_path

    if segmentation_results_path:                               # 추가
        task.segmentation_results_path = segmentation_results_path

    if segmentation_visualization_path:                         # 추가
        task.segmentation_visualization_path = segmentation_visualization_path

    if inpainted_panorama_path:                                 # 추가
        task.inpainted_panorama_path = inpainted_panorama_path

    if error_details:
        task.error_details = error_details

    return True
```

**예상 소요**: 10분
**의존성**: Task 1.1 완료
**검증**: 단위 테스트 또는 수동 호출 확인

---

### Phase 2: LangGraph 노드에서 실시간 업데이트 (핵심)

#### Task 2.1: WorkflowState에 task_id 추가

**파일**: `app/workflows/states.py`

**작업 내용**:
- `WorkflowState` TypedDict에 `task_id: str` 필드 추가

**변경 예시**:
```python
class WorkflowState(TypedDict):
    """Represents the shared state that flows through the LangGraph workflow."""

    task_id: str                                    # 추가
    user_input: str
    rewritten_query: str
    scene_name: str
    panorama_path: str
    segmentation_data: Dict[str, object]
    inpainted_panorama_path: str
    ply_path: str
    messages: Annotated[List[BaseMessage], operator.add]
```

**파일**: `app/services/panorama_service.py`

**작업 내용**:
- 초기 state 생성 시 `task_id` 전달

**변경 예시**:
```python
initial_state = {
    "task_id": task_id,                            # 추가
    "user_input": request.text,
    "rewritten_query": "",
    "scene_name": scene_name,
    "panorama_path": "",
    "segmentation_data": {},
    "messages": []
}
```

**예상 소요**: 10분
**의존성**: 없음
**검증**: 타입 체크 통과

---

#### Task 2.2: panorama_generation_node 수정

**파일**: `app/workflows/nodes.py:62-207`

**작업 내용**:
- `task_manager` import 추가
- 파노라마 생성 완료 시점(line 187 근처)에 업데이트 로직 추가

**변경 예시**:
```python
from ..services.task_manager import task_manager
from ..models.panorama import TaskStatus

def panorama_generation_node(state: WorkflowState) -> WorkflowState:
    """Generate a panorama using the DreamScene360 API and post-process the file path."""

    # ... 기존 코드 ...

    final_path = local_result_path or ""
    print(f"🎯 Final panorama path: {final_path}")

    # 실시간 업데이트 추가
    if final_path:
        task_manager.update_task_status(
            task_id=state["task_id"],
            status=TaskStatus.PROCESSING,
            message="Panorama generated, processing segmentation...",
            panorama_path=final_path
        )
        print(f"✅ Updated task_manager with panorama_path: {final_path}")

    return {
        **state,
        "panorama_path": final_path,
        "messages": [
            HumanMessage(
                content=f"Panorama generated: {final_path or result_path}"
            )
        ],
    }
```

**예상 소요**: 20분
**의존성**: Task 1.2, Task 2.1 완료
**검증**: 파노라마 생성 후 `/status/{task_id}` 호출해서 `panorama_path` 확인

---

#### Task 2.3: segmentation_node 수정

**파일**: `app/workflows/nodes.py:209-295`

**작업 내용**:
- 세그먼테이션 완료 시점(line 257 근처)에 업데이트 로직 추가

**변경 예시**:
```python
def segmentation_node(state: WorkflowState) -> WorkflowState:
    """Run segmentation on the generated panorama image."""

    # ... 기존 코드 ...

    print("✅ Segmentation completed")
    print(f"📋 Found objects: {list(result['segmentation_data'].get('prompts', {}).keys())}")

    # 실시간 업데이트 추가
    scene_name = state['scene_name']
    results_path = f"/home/0in/workspace/Text2VR/masking_output/{scene_name}/results.json"
    viz_path = f"/home/0in/workspace/Text2VR/masking_output/{scene_name}/visualizations/panorama_visualization.png"

    task_manager.update_task_status(
        task_id=state["task_id"],
        status=TaskStatus.PROCESSING,
        message="Segmentation completed, starting inpainting...",
        segmentation_results_path=results_path,
        segmentation_visualization_path=viz_path
    )
    print(f"✅ Updated task_manager with segmentation results")

    # Asset cropping with transparency
    # ... 기존 코드 ...
```

**예상 소요**: 15분
**의존성**: Task 1.2, Task 2.1 완료
**검증**: 세그먼테이션 완료 후 `/status/{task_id}` 호출

---

#### Task 2.4: inpainting_node 수정

**파일**: `app/workflows/nodes.py:298-379`

**작업 내용**:
- 인페인팅 완료 시점(line 363 근처)에 업데이트 로직 추가

**변경 예시**:
```python
def inpainting_node(state: WorkflowState) -> WorkflowState:
    """Run background inpainting on the panorama"""

    # ... 기존 코드 ...

    print(f"✅ Inpainting completed: {result_path}")

    # 컨테이너 경로를 호스트 경로로 변환
    host_result_path = result_path.replace(
        "/workspace/inpainted_pano/", "/home/0in/workspace/Text2VR/inpainted_pano/"
    )

    # 실시간 업데이트 추가
    task_manager.update_task_status(
        task_id=state["task_id"],
        status=TaskStatus.PROCESSING,
        message="Inpainting completed, generating PLY...",
        inpainted_panorama_path=host_result_path
    )
    print(f"✅ Updated task_manager with inpainted panorama: {host_result_path}")

    # 인페인팅 작업 완료 후 컨테이너 중지 (VRAM 절약)
    # ... 기존 코드 ...
```

**예상 소요**: 15분
**의존성**: Task 1.2, Task 2.1 완료
**검증**: 인페인팅 완료 후 `/status/{task_id}` 호출

---

### Phase 3: 백엔드 API 엔드포인트 추가

#### Task 3.1: 세그먼테이션 결과 제공 API

**파일**: `app/api/panorama.py`

**작업 내용**:
- 세그먼테이션 시각화 이미지 제공 엔드포인트 추가
- 세그먼테이션 JSON 메타데이터 제공 엔드포인트 추가

**변경 예시**:
```python
@router.get("/segmentation/{task_id}")
async def get_segmentation_visualization(task_id: str):
    """Download segmentation visualization image"""
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not task.segmentation_visualization_path:
        raise HTTPException(
            status_code=400,
            detail="Segmentation not yet available"
        )

    if not os.path.exists(task.segmentation_visualization_path):
        raise HTTPException(status_code=404, detail="Segmentation visualization file not found")

    return FileResponse(
        path=task.segmentation_visualization_path,
        filename=f"segmentation_{task.scene_name}.png",
        media_type="image/png"
    )


@router.get("/segmentation/{task_id}/json")
async def get_segmentation_json(task_id: str):
    """Get segmentation metadata (results.json)"""
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not task.segmentation_results_path:
        raise HTTPException(
            status_code=400,
            detail="Segmentation results not yet available"
        )

    if not os.path.exists(task.segmentation_results_path):
        raise HTTPException(status_code=404, detail="Segmentation results file not found")

    return FileResponse(
        path=task.segmentation_results_path,
        filename=f"segmentation_{task.scene_name}.json",
        media_type="application/json"
    )
```

**예상 소요**: 20분
**의존성**: Task 1.1, 1.2 완료
**검증**: curl 또는 브라우저로 직접 테스트
```bash
curl http://localhost:8000/segmentation/{task_id}
curl http://localhost:8000/segmentation/{task_id}/json
```

---

#### Task 3.2: 인페인팅 결과 제공 API

**파일**: `app/api/panorama.py`

**작업 내용**:
- 인페인팅된 파노라마 이미지 제공 엔드포인트 추가

**변경 예시**:
```python
@router.get("/inpainted/{task_id}")
async def get_inpainted_panorama(task_id: str):
    """Download inpainted panorama image"""
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not task.inpainted_panorama_path:
        raise HTTPException(
            status_code=400,
            detail="Inpainted panorama not yet available"
        )

    if not os.path.exists(task.inpainted_panorama_path):
        raise HTTPException(status_code=404, detail="Inpainted panorama file not found")

    return FileResponse(
        path=task.inpainted_panorama_path,
        filename=f"inpainted_{task.scene_name}.png",
        media_type="image/png"
    )
```

**예상 소요**: 15분
**의존성**: Task 1.1, 1.2 완료
**검증**: curl 테스트
```bash
curl http://localhost:8000/inpainted/{task_id} --output inpainted.png
```

---

### Phase 4: 프론트엔드 타입 및 API 서비스 확장

#### Task 4.1: TypeScript 인터페이스 확장

**파일**: `src/types/api.ts`

**작업 내용**:
- `StatusResponse` 인터페이스에 중간 결과물 필드 추가

**변경 예시**:
```typescript
export interface StatusResponse {
  status: 'queued' | 'processing' | 'completed' | 'failed';
  message: string;
  task_id: string;
  progress?: number;
  panorama_path?: string;                         // 추가
  segmentation_results_path?: string;             // 추가
  segmentation_visualization_path?: string;       // 추가
  inpainted_panorama_path?: string;               // 추가
  scene_name?: string;                            // 추가
}
```

**예상 소요**: 5분
**의존성**: 없음
**검증**: TypeScript 컴파일 통과 (`npm run build`)

---

#### Task 4.2: API 서비스 메서드 추가

**파일**: `src/services/apiService.ts`

**작업 내용**:
- 세그먼테이션 URL 생성 메서드 추가
- 인페인팅 URL 생성 메서드 추가

**변경 예시**:
```typescript
class ApiService {
  private baseUrl = '';

  // ... 기존 메서드 ...

  getSegmentationUrl(taskId: string): string {
    return `/segmentation/${taskId}?t=${Date.now()}`;
  }

  getSegmentationJsonUrl(taskId: string): string {
    return `/segmentation/${taskId}/json?t=${Date.now()}`;
  }

  getInpaintedUrl(taskId: string): string {
    return `/inpainted/${taskId}?t=${Date.now()}`;
  }
}
```

**예상 소요**: 5분
**의존성**: Task 4.1 완료
**검증**: 타입 체크 통과

---

### Phase 5: 프론트엔드 UI 실시간 표시

#### Task 5.1: 결과물 표시 컴포넌트 생성

**파일**: `src/components/ProgressiveResults.tsx` (신규)

**작업 내용**:
- 3가지 결과물을 단계별로 표시하는 컴포넌트 생성
- 각 섹션은 해당 경로가 있을 때만 렌더링
- 로딩 상태 표시

**구현 예시**:
```typescript
import React from 'react';

interface ProgressiveResultsProps {
  taskId: string | null;
  panoramaPath?: string;
  segmentationPath?: string;
  inpaintedPath?: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
}

export const ProgressiveResults: React.FC<ProgressiveResultsProps> = ({
  taskId,
  panoramaPath,
  segmentationPath,
  inpaintedPath,
  status
}) => {
  if (!taskId) return null;

  return (
    <div style={{ marginTop: '30px' }}>
      <h2>Pipeline Progress</h2>

      {/* 1. Panorama Section */}
      <div style={{ marginBottom: '30px' }}>
        <h3>
          1. Generated Panorama
          {panoramaPath ? ' ✅' : ' ⏳'}
        </h3>
        {panoramaPath ? (
          <img
            src={`/panorama/${taskId}?t=${Date.now()}`}
            alt="Generated panorama"
            style={{ maxWidth: '100%', borderRadius: '10px' }}
          />
        ) : (
          <p>Generating panorama...</p>
        )}
      </div>

      {/* 2. Segmentation Section */}
      <div style={{ marginBottom: '30px' }}>
        <h3>
          2. Segmentation Results
          {segmentationPath ? ' ✅' : ' ⏳'}
        </h3>
        {segmentationPath ? (
          <img
            src={`/segmentation/${taskId}?t=${Date.now()}`}
            alt="Segmentation visualization"
            style={{ maxWidth: '100%', borderRadius: '10px' }}
          />
        ) : panoramaPath ? (
          <p>Processing segmentation...</p>
        ) : (
          <p>Waiting for panorama...</p>
        )}
      </div>

      {/* 3. Inpainted Panorama Section */}
      <div style={{ marginBottom: '30px' }}>
        <h3>
          3. Inpainted Panorama
          {inpaintedPath ? ' ✅' : ' ⏳'}
        </h3>
        {inpaintedPath ? (
          <img
            src={`/inpainted/${taskId}?t=${Date.now()}`}
            alt="Inpainted panorama"
            style={{ maxWidth: '100%', borderRadius: '10px' }}
          />
        ) : segmentationPath ? (
          <p>Processing inpainting...</p>
        ) : (
          <p>Waiting for segmentation...</p>
        )}
      </div>
    </div>
  );
};
```

**예상 소요**: 30분
**의존성**: Task 4.1, 4.2 완료
**검증**: UI에서 시각적 확인

---

#### Task 5.2: App.tsx 폴링 로직 수정

**파일**: `src/App.tsx`

**작업 내용**:
- 상태 변수 추가 (각 결과물 경로)
- `startStatusChecking` 함수 수정하여 중간 결과물도 저장
- ProgressiveResults 컴포넌트 통합

**변경 예시**:
```typescript
export const App: React.FC = () => {
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showPanorama, setShowPanorama] = useState(false);

  // 중간 결과물 상태 추가
  const [panoramaPath, setPanoramaPath] = useState<string | null>(null);
  const [segmentationPath, setSegmentationPath] = useState<string | null>(null);
  const [inpaintedPath, setInpaintedPath] = useState<string | null>(null);

  const statusCheckInterval = useRef<NodeJS.Timeout | null>(null);

  const startStatusChecking = (taskId: string) => {
    if (statusCheckInterval.current) {
      clearInterval(statusCheckInterval.current);
    }

    statusCheckInterval.current = setInterval(async () => {
      try {
        const statusResponse = await apiService.getStatus(taskId);
        setStatus(statusResponse);

        // 중간 결과물 업데이트
        if (statusResponse.panorama_path) {
          setPanoramaPath(statusResponse.panorama_path);
        }
        if (statusResponse.segmentation_visualization_path) {
          setSegmentationPath(statusResponse.segmentation_visualization_path);
        }
        if (statusResponse.inpainted_panorama_path) {
          setInpaintedPath(statusResponse.inpainted_panorama_path);
        }

        if (statusResponse.status === 'completed') {
          stopStatusChecking();
          setShowPanorama(true);
          setIsLoading(false);
        } else if (statusResponse.status === 'failed') {
          stopStatusChecking();
          setIsLoading(false);
        }
      } catch (error) {
        console.error('Status check error:', error);
        // ... 에러 핸들링 ...
      }
    }, 2000);
  };

  const handleGenerate = async (text: string, sceneName: string) => {
    setIsLoading(true);
    setShowPanorama(false);
    setStatus(null);

    // 중간 결과물 초기화
    setPanoramaPath(null);
    setSegmentationPath(null);
    setInpaintedPath(null);

    // ... 기존 코드 ...
  };

  return (
    <div className="app">
      <div className="container">
        <h1 className="title">Text2VR</h1>

        <InputForm
          onSubmit={handleGenerate}
          isLoading={isLoading}
        />

        <StatusSection
          status={status}
          visible={!!status}
        />

        {/* 기존 VRPanoramaViewer 대신 ProgressiveResults 사용 */}
        <ProgressiveResults
          taskId={currentTaskId}
          panoramaPath={panoramaPath}
          segmentationPath={segmentationPath}
          inpaintedPath={inpaintedPath}
          status={status?.status || 'queued'}
        />
      </div>
    </div>
  );
};
```

**예상 소요**: 20분
**의존성**: Task 5.1 완료
**검증**: 전체 파이프라인 실행해서 단계별 표시 확인

---

### Phase 6: 통합 테스트 및 검증

#### Task 6.1: E2E 테스트

**작업 내용**:
- 전체 파이프라인 실행
- 각 단계마다 UI 업데이트 확인
- 타이밍 검증 (각 결과물이 생성 직후 표시되는지)

**테스트 시나리오**:
1. 프론트엔드에서 텍스트 입력 및 제출
2. 파노라마 생성 완료 → UI에 즉시 표시되는지 확인
3. 세그먼테이션 완료 → UI에 즉시 표시되는지 확인
4. 인페인팅 완료 → UI에 즉시 표시되는지 확인
5. 전체 파이프라인 완료 → 최종 상태 확인

**검증 포인트**:
- `/status/{task_id}` API 응답에 각 경로가 포함되는지
- 프론트엔드 상태가 2초마다 업데이트되는지
- 각 이미지가 올바르게 렌더링되는지

**예상 소요**: 30분
**의존성**: 모든 Phase 1-5 완료

---

#### Task 6.2: 에러 핸들링 보완

**작업 내용**:
- 중간 단계 실패 시 처리
- 파일 없을 때 fallback
- 타임아웃 처리

**구현 항목**:
1. 백엔드 API에서 파일 존재 확인 강화
2. 프론트엔드에서 이미지 로드 실패 처리
3. 폴링 타임아웃 설정 (예: 10분)

**변경 예시**:
```typescript
// 프론트엔드 이미지 로드 에러 핸들링
<img
  src={`/segmentation/${taskId}`}
  alt="Segmentation"
  onError={(e) => {
    console.error('Failed to load segmentation image');
    e.currentTarget.style.display = 'none';
  }}
/>
```

**예상 소요**: 20분
**의존성**: Task 6.1 완료

---

## 📊 구현 계획 요약

| Phase | Task 수 | 예상 소요 | 우선순위 | 파일 수 |
|-------|---------|----------|---------|---------|
| Phase 1: 데이터 모델 | 2 | 25분 | P0 (필수) | 2 |
| Phase 2: LangGraph 업데이트 | 4 | 60분 | P0 (필수) | 3 |
| Phase 3: 백엔드 API | 2 | 35분 | P0 (필수) | 1 |
| Phase 4: 프론트 타입/API | 2 | 10분 | P0 (필수) | 2 |
| Phase 5: 프론트 UI | 2 | 50분 | P0 (필수) | 2 |
| Phase 6: 테스트 | 2 | 50분 | P1 (권장) | - |
| **총계** | **14 tasks** | **~230분 (3.8시간)** | | **10 파일** |

### 수정 파일 목록

**백엔드** (5 파일):
1. `app/models/panorama.py` - 데이터 모델
2. `app/services/task_manager.py` - Task 관리
3. `app/workflows/states.py` - LangGraph state
4. `app/workflows/nodes.py` - LangGraph 노드
5. `app/api/panorama.py` - API 엔드포인트

**프론트엔드** (4 파일):
1. `src/types/api.ts` - TypeScript 타입
2. `src/services/apiService.ts` - API 서비스
3. `src/components/ProgressiveResults.tsx` - 신규 컴포넌트
4. `src/App.tsx` - 메인 앱

---

## 🚀 구현 전략

### 권장 접근법: Bottom-up (데이터 모델 → 백엔드 → 프론트)

1. **Phase 1 → Phase 2 → Phase 3** (백엔드 완성)
2. **Phase 4 → Phase 5** (프론트엔드 완성)
3. **Phase 6** (검증)

### 대안: Incremental (한 결과물씩 E2E)

1. 파노라마만 E2E 구현
2. 세그먼테이션 추가
3. 인페인팅 추가

---

## 🚨 잠재적 리스크 및 고려사항

### 1. LangGraph State 전달

**리스크**: `task_id`를 state에 넣는 것이 LangGraph 아키텍처와 맞는지 검토 필요

**완화 방안**:
- LangGraph `WorkflowState`는 TypedDict이므로 필드 추가는 안전
- State는 각 노드 간 전달되므로 `task_id` 접근 가능

### 2. 동시성 이슈

**리스크**: task_manager 딕셔너리 업데이트 시 thread-safety

**현재 상황**:
- FastAPI background tasks는 단일 워커 가정
- 현재 구현은 딕셔너리 직접 수정

**완화 방안**:
- 프로덕션 환경에서는 Redis 등 외부 스토어 사용 고려
- 또는 threading.Lock 사용

### 3. 파일 경로 불일치

**리스크**: Docker 컨테이너 경로 vs 호스트 경로 변환 로직 오류

**완화 방안**:
- 경로 변환 로직 철저히 테스트
- 로깅 강화하여 실제 경로 추적

### 4. 파일 생성 타이밍

**리스크**: 파일이 생성되기 전에 API 요청이 올 수 있음

**완화 방안**:
- 백엔드 API에서 파일 존재 확인
- 404 에러 대신 "Not yet available" 메시지 반환
- 프론트엔드에서 graceful fallback

---

## ✅ 성공 기준

1. **기능적 요구사항**:
   - ✅ 파노라마 생성 즉시 UI 표시
   - ✅ 세그먼테이션 완료 즉시 UI 표시
   - ✅ 인페인팅 완료 즉시 UI 표시

2. **성능 요구사항**:
   - ✅ 각 결과물 생성 후 2초 이내 UI 업데이트
   - ✅ 폴링 오버헤드 최소화

3. **안정성 요구사항**:
   - ✅ 중간 단계 실패 시 에러 핸들링
   - ✅ 파일 누락 시 적절한 메시지 표시

---

## 📝 다음 단계

1. ✅ 이 문서 검토 및 승인
2. Phase 1부터 순차적 구현 시작
3. 각 Phase 완료 후 중간 검증
4. Phase 6 완료 후 최종 테스트
5. 프로덕션 배포

---

**문서 버전**: 1.0
**최종 수정**: 2025-10-21

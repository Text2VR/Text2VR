# Real-Time UI Update Implementation Plan

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
````

**Date**: 2025-10-21
**Goal**: Display intermediate outputs from the LangGraph pipeline on the frontend UI in real time.

---

## 📋 Requirements Clarification

### Core Requirement

> Three intermediate results produced during the LangGraph pipeline must be displayed on the UI **as soon as each one is generated**.

#### The 3 Intermediate Outputs

1. **Panorama** (`panorama.png`)
2. **Segmentation outputs** (masks, visualization image)
3. **Inpainted panorama** (`inpainted_panorama.png`)

#### Current Issues

* ❌ UI is updated only after the entire pipeline finishes.
* ❌ Segmentation and inpainting outputs either do not appear on the UI or display incorrectly.

#### Target State

* ✅ When panorama generation finishes → show immediately on UI.
* ✅ When segmentation finishes → show segmentation results immediately (masks/visualization).
* ✅ When inpainting finishes → show inpainted panorama immediately.

---

## 🔍 Root Cause Analysis

### Three Main Root Causes

#### 1. LangGraph pipeline runs synchronously

**Location**: `app/services/panorama_service.py:159`

```python
result = workflow.invoke(initial_state)  # Synchronous execution - blocking until all nodes finish
```

* **Problem**: LangGraph’s `workflow.invoke()` is a **synchronous** function. It blocks until all nodes
  (query_rewrite → panorama_generation → segmentation → inpainting → ply_generation)
  complete sequentially.
* **Consequence**: Even if the panorama, segmentation, and inpainting outputs are generated in the middle,
  the function does not return until the entire pipeline finishes, so **UI updates cannot happen earlier**.

#### 2. Intermediate results are not written into `task_manager`

**Location**: `app/workflows/nodes.py`

* **Panorama generation node** (lines 62–207): after generating the panorama, the result is stored only in `state`, not in `task_manager`.
* **Segmentation node** (lines 209–295): result stored only in `state`, not in `task_manager`.
* **Inpainting node** (lines 298–379): result stored only in `state`, not in `task_manager`.

**Current logic**:

```python
# Example from panorama_generation_node
return {
    **state,
    "panorama_path": final_path,  # stored only in state
    # no task_manager.update_task_status(...) call!
}
```

**Verification**:

```bash
$ grep "task_manager" Text2VR/app/workflows/nodes.py
# Result: No matches found
```

**Consequence**:
Even if the frontend calls `/status/{task_id}`, `task_manager` does not know the paths to intermediate outputs, so the UI has nothing to display.

#### 3. Backend APIs do not expose intermediate outputs

**Current status**:

* `app/api/panorama.py` – only exposes `/panorama/{task_id}` which returns the final panorama.
* No endpoint for segmentation outputs.
* No endpoint for inpainted panorama.

**Frontend status interface**:

```typescript
// src/types/api.ts:11-16
export interface StatusResponse {
  status: 'queued' | 'processing' | 'completed' | 'failed';
  message: string;
  task_id: string;
  progress?: number;
  // No panorama_path, segmentation_data, or inpainted_panorama_path fields!
}
```

**Data model**:

```python
# app/models/panorama.py:48-58
class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    message: str
    panorama_path: Optional[str] = None          # ✅ exists
    scene_name: Optional[str] = None
    # segmentation_data_path: missing             # ❌
    # inpainted_panorama_path: missing            # ❌
```

**Consequence**:
The frontend has no way to receive intermediate outputs.

---

## 🏗️ Architectural Solution

### Current Data Flow

```
Frontend (polling /status)
    ↓ (every 2s)
    └─→ Backend: GET /status/{task_id}
            ↓
            └─→ task_manager.get_task()
                    ↓
                    └─→ TaskInfo returned (only panorama_path; no intermediate outputs)

LangGraph Pipeline (blocking):
query_rewrite → panorama_gen → segmentation → inpainting → ply_gen
                     ↓              ↓               ↓
                  (stored only   (stored only   (stored only
                   in state)     in state)      in state)
                                                   ↓
                                            Only after final step
                                            task_manager is updated
```

### Improved Data Flow

```
Frontend (polling /status)
    ↓ (every 2s)
    └─→ Backend: GET /status/{task_id}
            ↓
            └─→ task_manager.get_task()
                    ↓
                    └─→ StatusResponse {
                          panorama_path: "...",             ✅
                          segmentation_path: "...",         ✅
                          inpainted_panorama_path: "..."    ✅
                        }

LangGraph Pipeline:
query_rewrite → panorama_gen → segmentation → inpainting → ply_gen
                     ↓              ↓               ↓
                  state +        state +        state +
                  task_manager   task_manager   task_manager
                  ✅ real-time   ✅ real-time   ✅ real-time
                     updates        updates        updates
```

---

## 📝 Detailed Tasks

### Phase 1: Extend backend data models (required foundation)

#### Task 1.1: Extend Pydantic models

**File**: `app/models/panorama.py`

**Work**:

* Add fields to `TaskInfo`:

  * `segmentation_results_path: Optional[str] = None` – path to JSON
  * `segmentation_visualization_path: Optional[str] = None` – path to visualization image
  * `inpainted_panorama_path: Optional[str] = None`
* Add the same fields to `StatusResponse`.

**Example change**:

```python
class TaskInfo(BaseModel):
    """Internal task information"""
    task_id: str
    status: TaskStatus
    message: str
    panorama_path: Optional[str] = None
    segmentation_results_path: Optional[str] = None              # new
    segmentation_visualization_path: Optional[str] = None        # new
    inpainted_panorama_path: Optional[str] = None                # new
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
    segmentation_results_path: Optional[str] = None              # new
    segmentation_visualization_path: Optional[str] = None        # new
    inpainted_panorama_path: Optional[str] = None                # new
    scene_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
```

**Estimated time**: 15 min
**Dependencies**: None
**Validation**: Pydantic validation passes.

---

#### Task 1.2: Update TaskManager

**File**: `app/services/task_manager.py`

**Work**:

* Extend `update_task_status()` parameters:

  * `segmentation_results_path: Optional[str] = None`
  * `segmentation_visualization_path: Optional[str] = None`
  * `inpainted_panorama_path: Optional[str] = None`
* Implement logic to update each field.
* Always update `updated_at`.

**Example change**:

```python
def update_task_status(
    self,
    task_id: str,
    status: TaskStatus,
    message: str,
    panorama_path: Optional[str] = None,
    segmentation_results_path: Optional[str] = None,           # new
    segmentation_visualization_path: Optional[str] = None,     # new
    inpainted_panorama_path: Optional[str] = None,             # new
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

    if segmentation_results_path:                               # new
        task.segmentation_results_path = segmentation_results_path

    if segmentation_visualization_path:                         # new
        task.segmentation_visualization_path = segmentation_visualization_path

    if inpainted_panorama_path:                                 # new
        task.inpainted_panorama_path = inpainted_panorama_path

    if error_details:
        task.error_details = error_details

    return True
```

**Estimated time**: 10 min
**Dependencies**: Task 1.1
**Validation**: Unit test or manual call.

---

### Phase 2: Real-time updates from LangGraph nodes (core)

#### Task 2.1: Add `task_id` to `WorkflowState`

**File**: `app/workflows/states.py`

**Work**:

* Add `task_id: str` to `WorkflowState` TypedDict.

**Example change**:

```python
class WorkflowState(TypedDict):
    """Represents the shared state that flows through the LangGraph workflow."""

    task_id: str                                    # new
    user_input: str
    rewritten_query: str
    scene_name: str
    panorama_path: str
    segmentation_data: Dict[str, object]
    inpainted_panorama_path: str
    ply_path: str
    messages: Annotated[List[BaseMessage], operator.add]
```

**File**: `app/services/panorama_service.py`

**Work**:

* Include `task_id` when building the initial state.

**Example change**:

```python
initial_state = {
    "task_id": task_id,                            # new
    "user_input": request.text,
    "rewritten_query": "",
    "scene_name": scene_name,
    "panorama_path": "",
    "segmentation_data": {},
    "messages": []
}
```

**Estimated time**: 10 min
**Dependencies**: None
**Validation**: Type checks.

---

#### Task 2.2: Update `panorama_generation_node`

**File**: `app/workflows/nodes.py:62-207`

**Work**:

* Import `task_manager`.
* After panorama is generated (around line 187), update `task_manager`.

**Example**:

```python
from ..services.task_manager import task_manager
from ..models.panorama import TaskStatus

def panorama_generation_node(state: WorkflowState) -> WorkflowState:
    """Generate a panorama using the DreamScene360 API and post-process the file path."""

    # ... existing code ...

    final_path = local_result_path or ""
    print(f"🎯 Final panorama path: {final_path}")

    # Real-time update
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

**Estimated time**: 20 min
**Dependencies**: Tasks 1.2, 2.1
**Validation**: After panorama generation, call `/status/{task_id}` and check `panorama_path`.

---

#### Task 2.3: Update `segmentation_node`

**File**: `app/workflows/nodes.py:209-295`

**Work**:

* After segmentation completes (around line 257), update `task_manager`.

**Example**:

```python
def segmentation_node(state: WorkflowState) -> WorkflowState:
    """Run segmentation on the generated panorama image."""

    # ... existing code ...

    print("✅ Segmentation completed")
    print(f"📋 Found objects: {list(result['segmentation_data'].get('prompts', {}).keys())}")

    # Real-time update
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
    print("✅ Updated task_manager with segmentation results")

    # Asset cropping with transparency
    # ... existing code ...
```

**Estimated time**: 15 min
**Dependencies**: Tasks 1.2, 2.1
**Validation**: After segmentation, call `/status/{task_id}`.

---

#### Task 2.4: Update `inpainting_node`

**File**: `app/workflows/nodes.py:298-379`

**Work**:

* After inpainting completes (around line 363), update `task_manager`.

**Example**:

```python
def inpainting_node(state: WorkflowState) -> WorkflowState:
    """Run background inpainting on the panorama"""

    # ... existing code ...

    print(f"✅ Inpainting completed: {result_path}")

    # Convert container path to host path
    host_result_path = result_path.replace(
        "/workspace/inpainted_pano/", "/home/0in/workspace/Text2VR/inpainted_pano/"
    )

    # Real-time update
    task_manager.update_task_status(
        task_id=state["task_id"],
        status=TaskStatus.PROCESSING,
        message="Inpainting completed, generating PLY...",
        inpainted_panorama_path=host_result_path
    )
    print(f"✅ Updated task_manager with inpainted panorama: {host_result_path}")

    # Stop container after inpainting (VRAM savings)
    # ... existing code ...
```

**Estimated time**: 15 min
**Dependencies**: Tasks 1.2, 2.1
**Validation**: After inpainting, call `/status/{task_id}`.

---

### Phase 3: Backend API endpoints for intermediate outputs

#### Task 3.1: Segmentation result APIs

**File**: `app/api/panorama.py`

**Work**:

* Add endpoint to return segmentation visualization image.
* Add endpoint to return segmentation JSON metadata.

**Example**:

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

**Estimated time**: 20 min
**Dependencies**: Tasks 1.1, 1.2
**Validation**:

```bash
curl http://localhost:8000/segmentation/{task_id}
curl http://localhost:8000/segmentation/{task_id}/json
```

---

#### Task 3.2: Inpainted panorama API

**File**: `app/api/panorama.py`

**Work**:

* Add endpoint to serve the inpainted panorama.

**Example**:

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

**Estimated time**: 15 min
**Dependencies**: Tasks 1.1, 1.2
**Validation**:

```bash
curl http://localhost:8000/inpainted/{task_id} --output inpainted.png
```

---

### Phase 4: Frontend types & API service

#### Task 4.1: Extend TypeScript interfaces

**File**: `src/types/api.ts`

**Work**:

* Add fields for intermediate outputs to `StatusResponse`.

**Example**:

```typescript
export interface StatusResponse {
  status: 'queued' | 'processing' | 'completed' | 'failed';
  message: string;
  task_id: string;
  progress?: number;
  panorama_path?: string;                         // new
  segmentation_results_path?: string;             // new
  segmentation_visualization_path?: string;       // new
  inpainted_panorama_path?: string;               // new
  scene_name?: string;                            // new
}
```

**Estimated time**: 5 min
**Dependencies**: None
**Validation**: `npm run build`.

---

#### Task 4.2: Extend API service

**File**: `src/services/apiService.ts`

**Work**:

* Add helpers to build URLs for segmentation and inpainted panoramas.

**Example**:

```typescript
class ApiService {
  private baseUrl = '';

  // ... existing methods ...

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

**Estimated time**: 5 min
**Dependencies**: Task 4.1
**Validation**: Type checks.

---

### Phase 5: Frontend UI for real-time display

#### Task 5.1: Progressive results component

**File**: `src/components/ProgressiveResults.tsx` (new)

**Work**:

* Create a component that displays the three outputs step by step.
* Each section should render only when its path is available.
* Show loading states when not yet available.

**Example**:

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

**Estimated time**: 30 min
**Dependencies**: Tasks 4.1, 4.2
**Validation**: Visual check in UI.

---

#### Task 5.2: Update polling logic in `App.tsx`

**File**: `src/App.tsx`

**Work**:

* Add state variables for each intermediate path.
* Modify `startStatusChecking` to store intermediate paths.
* Integrate `ProgressiveResults` component.

**Example**:

```typescript
export const App: React.FC = () => {
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showPanorama, setShowPanorama] = useState(false);

  // Intermediate result states
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

        // Update intermediate paths
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
        // ... error handling ...
      }
    }, 2000);
  };

  const handleGenerate = async (text: string, sceneName: string) => {
    setIsLoading(true);
    setShowPanorama(false);
    setStatus(null);

    // Reset intermediate states
    setPanoramaPath(null);
    setSegmentationPath(null);
    setInpaintedPath(null);

    // ... existing code ...
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

        <ProgressiveResults
          taskId={currentTaskId}
          panoramaPath={panoramaPath || undefined}
          segmentationPath={segmentationPath || undefined}
          inpaintedPath={inpaintedPath || undefined}
          status={status?.status || 'queued'}
        />
      </div>
    </div>
  );
};
```

**Estimated time**: 20 min
**Dependencies**: Task 5.1
**Validation**: Run full pipeline and verify step-by-step display.

---

### Phase 6: Integration testing and validation

#### Task 6.1: End-to-end test

**Work**:

* Run the entire pipeline.
* Check UI updates at each stage.
* Verify timing (each output appears shortly after it is generated).

**Test scenario**:

1. Submit a text prompt from the frontend.
2. After panorama generation → confirm it appears on UI.
3. After segmentation → confirm segmentation visualization appears.
4. After inpainting → confirm inpainted panorama appears.
5. At the end → check final status.

**Validation points**:

* `/status/{task_id}` includes all paths.
* Frontend updates every 2 seconds.
* Images render correctly.

**Estimated time**: 30 min
**Dependencies**: Phases 1–5.

---

#### Task 6.2: Error handling improvements

**Work**:

* Handle failures in intermediate steps.
* Handle missing files gracefully.
* Add timeout handling.

**Checklist**:

1. Strengthen file-existence checks in backend APIs.
2. Handle image load errors in the frontend.
3. Add a polling timeout (e.g., 10 minutes).

**Example**:

```typescript
// Frontend image load error handling
<img
  src={`/segmentation/${taskId}`}
  alt="Segmentation"
  onError={(e) => {
    console.error('Failed to load segmentation image');
    e.currentTarget.style.display = 'none';
  }}
/>
```

**Estimated time**: 20 min
**Dependencies**: Task 6.1.

---

## 📊 Implementation Plan Summary

| Phase                       | # Tasks      | Est. Time                 | Priority         | # Files      |
| --------------------------- | ------------ | ------------------------- | ---------------- | ------------ |
| Phase 1: Data models        | 2            | 25 min                    | P0 (required)    | 2            |
| Phase 2: LangGraph updates  | 4            | 60 min                    | P0 (required)    | 3            |
| Phase 3: Backend APIs       | 2            | 35 min                    | P0 (required)    | 1            |
| Phase 4: Frontend types/API | 2            | 10 min                    | P0 (required)    | 2            |
| Phase 5: Frontend UI        | 2            | 50 min                    | P0 (required)    | 2            |
| Phase 6: Testing            | 2            | 50 min                    | P1 (recommended) | -            |
| **Total**                   | **14 tasks** | **~230 min (~3.8 hours)** |                  | **10 files** |

### Files to be modified

**Backend** (5 files):

1. `app/models/panorama.py` – data models
2. `app/services/task_manager.py` – task management
3. `app/workflows/states.py` – LangGraph state
4. `app/workflows/nodes.py` – LangGraph nodes
5. `app/api/panorama.py` – API endpoints

**Frontend** (4 files):

1. `src/types/api.ts` – TypeScript types
2. `src/services/apiService.ts` – API service
3. `src/components/ProgressiveResults.tsx` – new component
4. `src/App.tsx` – main app

---

## 🚀 Implementation Strategy

### Recommended approach: Bottom-up (models → backend → frontend)

1. Implement **Phase 1 → Phase 2 → Phase 3** (backend complete).
2. Implement **Phase 4 → Phase 5** (frontend complete).
3. Run **Phase 6** (validation).

### Alternative: Incremental (E2E per output)

1. Implement E2E just for panorama first.
2. Add segmentation.
3. Add inpainting.

---

## 🚨 Risks and Considerations

### 1. LangGraph state design

**Risk**: Adding `task_id` to state may conflict with LangGraph design.

**Mitigation**:

* `WorkflowState` is a TypedDict, so adding fields is safe.
* State is passed between all nodes, so each node can access `task_id`.

---

### 2. Concurrency issues

**Risk**: `task_manager` uses an in-memory dict; updates may not be thread-safe.

**Current situation**:

* FastAPI background tasks assume a single worker in this setup.
* Current implementation modifies the dict directly.

**Mitigation**:

* For production, consider Redis or another external store.
* Or use a `threading.Lock` around updates.

---

### 3. Path mismatches

**Risk**: Errors when converting container paths to host paths.

**Mitigation**:

* Thoroughly test path conversion logic.
* Add logging for actual paths used.

---

### 4. File creation timing

**Risk**: API may be called before the file is physically written.

**Mitigation**:

* Backend API checks file existence.
* Return “Not yet available” instead of 404 when appropriate.
* Frontend handles this with graceful fallback.

---

## ✅ Success Criteria

1. **Functional**:

   * ✅ Panorama appears on UI immediately after generation.
   * ✅ Segmentation appears on UI immediately after completion.
   * ✅ Inpainted panorama appears on UI immediately after completion.

2. **Performance**:

   * ✅ Each output is reflected on the UI within 2 seconds of being generated.
   * ✅ Polling overhead is minimized.

3. **Stability**:

   * ✅ Errors in intermediate steps are handled cleanly.
   * ✅ Missing files result in clear messages, not crashes.

---

## 📝 Next Steps

1. ✅ Review and approve this document.
2. Start implementation from **Phase 1** in order.
3. Run intermediate validation after each phase.
4. Complete **Phase 6** and finalize E2E tests.
5. Deploy to production.

---

**Document Version**: 1.0
**Last Updated**: 2025-10-21

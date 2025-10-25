export interface GenerateRequest {
  text: string;
  scene_name?: string | null;
}

export interface GenerateResponse {
  task_id: string;
  message: string;
}

export interface StatusResponse {
  status: 'queued' | 'processing' | 'completed' | 'failed';
  message: string;
  task_id: string;
  progress?: number;
  panorama_path?: string;
  segmentation_results_path?: string;
  segmentation_visualization_path?: string;
  inpainted_panorama_path?: string;
  scene_name?: string;
}

export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  service: string;
  version: string;
}
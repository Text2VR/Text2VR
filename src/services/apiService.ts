import { GenerateRequest, GenerateResponse, StatusResponse, HealthResponse } from '../types/api';

class ApiService {
  private baseUrl = '';

  async generatePanorama(request: GenerateRequest): Promise<GenerateResponse> {
    const response = await fetch('/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async getStatus(taskId: string): Promise<StatusResponse> {
    const response = await fetch(`/status/${taskId}`);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  getPanoramaUrl(taskId: string): string {
    return `/panorama/${taskId}?t=${Date.now()}`;
  }

  getSegmentationUrl(taskId: string): string {
    return `/segmentation/${taskId}?t=${Date.now()}`;
  }

  getInpaintedUrl(taskId: string): string {
    return `/inpainted/${taskId}?t=${Date.now()}`;
  }

  async checkHealth(): Promise<HealthResponse> {
    const response = await fetch('/health');

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }
}

export const apiService = new ApiService();
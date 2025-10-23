import { GenerateRequest, GenerateResponse, StatusResponse, HealthResponse } from '../types/api';

class ApiService {
  private baseUrl = '';

  async generatePanorama(request: GenerateRequest): Promise<GenerateResponse> {
    console.log('[API] Sending generate request:', request);

    const response = await fetch('/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    console.log('[API] Generate response status:', response.status);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('[API] Generate error response:', errorText);
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('[API] Generate response data:', data);
    return data;
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

  async getSegmentationAssets(taskId: string): Promise<Array<{name: string, url: string}>> {
    const response = await fetch(`/segmentation/${taskId}`);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.assets;
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
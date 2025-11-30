import { GenerateRequest, GenerateResponse, StatusResponse, HealthResponse, SegmentedAsset } from '../types/api';

class ApiService {
  async generatePanorama(request: GenerateRequest): Promise<GenerateResponse> {
    const response = await fetch('/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('[API] Generate error:', errorText);
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

  async getSegmentationAssets(taskId: string): Promise<SegmentedAsset[]> {
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

  // New methods for 3D assets and PLY downloads
  async download3DAssets(sceneName?: string): Promise<Blob> {
    const url = sceneName
      ? `/unity/${sceneName}/assets.zip`
      : '/unity/latest/assets.zip';

    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Download failed: ${response.status}`);
    }

    return response.blob();
  }

  async downloadPLY(sceneName?: string): Promise<Blob> {
    const url = sceneName
      ? `/unity/${sceneName}/scene.ply`
      : '/unity/latest/scene.ply';

    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Download failed: ${response.status}`);
    }

    return response.blob();
  }

  get3DAssetUrl(sceneName: string, assetName: string): string {
    return `/unity/${sceneName}/assets/${assetName}.glb`;
  }
}

export const apiService = new ApiService();

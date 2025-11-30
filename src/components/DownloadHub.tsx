import React from 'react';

interface DownloadHubProps {
  taskId: string | null;
  panoramaReady: boolean;
  assetsReady: boolean;
  plyReady: boolean;
  sceneName?: string;
}

export const DownloadHub: React.FC<DownloadHubProps> = ({
  taskId,
  panoramaReady,
  assetsReady,
  plyReady,
  sceneName,
}) => {
  const handleDownload = async (type: 'panorama' | 'assets' | 'ply') => {
    if (!taskId) return;

    let url = '';
    let filename = '';

    switch (type) {
      case 'panorama':
        url = `/panorama/${taskId}`;
        filename = `${sceneName || taskId}_panorama.png`;
        break;
      case 'assets':
        url = `/unity/latest/assets.zip`;
        filename = `${sceneName || taskId}_3d_assets.zip`;
        break;
      case 'ply':
        url = sceneName ? `/unity/${sceneName}/scene.ply` : `/unity/latest/scene.ply`;
        filename = `${sceneName || taskId}_scene.ply`;
        break;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error('Download failed');

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(downloadUrl);
    } catch (error) {
      console.error('Download error:', error);
    }
  };

  return (
    <div className="download-hub">
      <h3 className="download-hub-title">Downloads</h3>
      <div className="download-grid">
        <button
          className="download-btn"
          onClick={() => handleDownload('panorama')}
          disabled={!panoramaReady}
        >
          <svg className="download-btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <path d="M21 15l-5-5L5 21" />
          </svg>
          <span>Panorama</span>
          <span style={{ opacity: 0.6, fontSize: '10px' }}>PNG</span>
        </button>

        <button
          className="download-btn"
          onClick={() => handleDownload('assets')}
          disabled={!assetsReady}
        >
          <svg className="download-btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
            <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
            <line x1="12" y1="22.08" x2="12" y2="12" />
          </svg>
          <span>3D Assets</span>
          <span style={{ opacity: 0.6, fontSize: '10px' }}>GLB (ZIP)</span>
        </button>

        <button
          className="download-btn"
          onClick={() => handleDownload('ply')}
          disabled={!plyReady}
        >
          <svg className="download-btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <circle cx="12" cy="12" r="3" />
            <path d="M12 2v3" />
            <path d="M12 19v3" />
            <path d="M2 12h3" />
            <path d="M19 12h3" />
          </svg>
          <span>Point Cloud</span>
          <span style={{ opacity: 0.6, fontSize: '10px' }}>PLY</span>
        </button>
      </div>
    </div>
  );
};

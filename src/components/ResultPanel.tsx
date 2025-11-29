import React, { useState, useEffect, useRef } from 'react';
import { apiService } from '../services/apiService';
import 'aframe';

type ViewerTab = 'panorama' | 'segmentation' | 'inpainted';

interface ResultPanelProps {
  taskId: string | null;
  panoramaPath?: string;
  segmentationPath?: string;
  inpaintedPath?: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | null;
}

interface SegmentedAsset {
  name: string;
  url: string;
}

export const ResultPanel: React.FC<ResultPanelProps> = ({
  taskId,
  panoramaPath,
  segmentationPath,
  inpaintedPath,
  status,
}) => {
  const [activeTab, setActiveTab] = useState<ViewerTab>('panorama');
  const [segmentedAssets, setSegmentedAssets] = useState<SegmentedAsset[]>([]);
  const [loadingAssets, setLoadingAssets] = useState(false);

  // Separate refs for each viewer to prevent conflicts
  const panoramaViewerRef = useRef<HTMLDivElement>(null);
  const inpaintedViewerRef = useRef<HTMLDivElement>(null);

  // Cache URLs to prevent flickering
  const [cachedPanoramaUrl, setCachedPanoramaUrl] = useState<string | null>(null);
  const [cachedInpaintedUrl, setCachedInpaintedUrl] = useState<string | null>(null);
  const [cachedAssetUrls, setCachedAssetUrls] = useState<Record<string, string>>({});

  // Update cached URLs only when path first becomes available
  useEffect(() => {
    if (panoramaPath && taskId && !cachedPanoramaUrl) {
      setCachedPanoramaUrl(apiService.getPanoramaUrl(taskId));
    }
  }, [panoramaPath, taskId, cachedPanoramaUrl]);

  useEffect(() => {
    if (inpaintedPath && taskId && !cachedInpaintedUrl) {
      setCachedInpaintedUrl(apiService.getInpaintedUrl(taskId));
    }
  }, [inpaintedPath, taskId, cachedInpaintedUrl]);

  // Reset everything when taskId changes (new generation)
  useEffect(() => {
    setCachedPanoramaUrl(null);
    setCachedInpaintedUrl(null);
    setCachedAssetUrls({});
    setSegmentedAssets([]);

    // Clear A-Frame scenes
    if (panoramaViewerRef.current) {
      panoramaViewerRef.current.innerHTML = '';
    }
    if (inpaintedViewerRef.current) {
      inpaintedViewerRef.current.innerHTML = '';
    }
  }, [taskId]);

  // Fetch segmented assets when available
  useEffect(() => {
    if (taskId && segmentationPath) {
      setLoadingAssets(true);
      apiService.getSegmentationAssets(taskId)
        .then(assets => {
          setSegmentedAssets(assets);
          const urls: Record<string, string> = {};
          assets.forEach(asset => {
            urls[asset.name] = `${asset.url}?t=${Date.now()}`;
          });
          setCachedAssetUrls(urls);
          setLoadingAssets(false);
        })
        .catch(err => {
          console.error('Failed to load segmented assets:', err);
          setLoadingAssets(false);
        });
    }
  }, [taskId, segmentationPath]);

  // Setup Panorama A-Frame viewer
  useEffect(() => {
    if (!panoramaViewerRef.current || !cachedPanoramaUrl) return;

    panoramaViewerRef.current.innerHTML = `
      <a-scene
        embedded
        style="width: 100%; height: 100%;"
        vr-mode-ui="enabled: true"
        background="color: #f1f5f9"
      >
        <a-sky
          src="${cachedPanoramaUrl}"
          rotation="0 -130 0"
        ></a-sky>
        <a-entity id="cameraRig" position="0 1.6 0">
          <a-camera
            look-controls="enabled: true"
            wasd-controls="enabled: false"
            position="0 0 0"
          ></a-camera>
        </a-entity>
      </a-scene>
    `;
  }, [cachedPanoramaUrl]);

  // Setup Inpainted A-Frame viewer (separate instance)
  useEffect(() => {
    if (!inpaintedViewerRef.current || !cachedInpaintedUrl) return;

    inpaintedViewerRef.current.innerHTML = `
      <a-scene
        embedded
        style="width: 100%; height: 100%;"
        vr-mode-ui="enabled: true"
        background="color: #f1f5f9"
      >
        <a-sky
          src="${cachedInpaintedUrl}"
          rotation="0 -130 0"
        ></a-sky>
        <a-entity id="cameraRig2" position="0 1.6 0">
          <a-camera
            look-controls="enabled: true"
            wasd-controls="enabled: false"
            position="0 0 0"
          ></a-camera>
        </a-entity>
      </a-scene>
    `;
  }, [cachedInpaintedUrl]);

  const getTabIndicatorStatus = (tab: ViewerTab): 'ready' | 'processing' | 'pending' => {
    switch (tab) {
      case 'panorama':
        return panoramaPath ? 'ready' : status === 'processing' ? 'processing' : 'pending';
      case 'segmentation':
        return segmentationPath ? 'ready' : panoramaPath ? 'processing' : 'pending';
      case 'inpainted':
        return inpaintedPath ? 'ready' : segmentationPath ? 'processing' : 'pending';
    }
  };

  return (
    <div className="result-panel">
      {/* Tab Navigation */}
      <div className="tab-nav">
        <button
          className={`tab-btn ${activeTab === 'panorama' ? 'active' : ''}`}
          onClick={() => setActiveTab('panorama')}
        >
          <span className={`tab-btn-indicator ${getTabIndicatorStatus('panorama')}`} />
          Panorama
        </button>
        <button
          className={`tab-btn ${activeTab === 'segmentation' ? 'active' : ''}`}
          onClick={() => setActiveTab('segmentation')}
        >
          <span className={`tab-btn-indicator ${getTabIndicatorStatus('segmentation')}`} />
          Segmentation
          {segmentedAssets.length > 0 && (
            <span className="tab-count">{segmentedAssets.length}</span>
          )}
        </button>
        <button
          className={`tab-btn ${activeTab === 'inpainted' ? 'active' : ''}`}
          onClick={() => setActiveTab('inpainted')}
        >
          <span className={`tab-btn-indicator ${getTabIndicatorStatus('inpainted')}`} />
          Inpainted
        </button>
      </div>

      {/* Viewer Containers - All rendered but only active one visible */}
      <div className="viewer-wrapper">
        {/* Panorama Viewer */}
        <div
          className={`viewer-container ${activeTab === 'panorama' ? 'active' : 'hidden'}`}
        >
          <div ref={panoramaViewerRef} style={{ width: '100%', height: '100%' }} />
          {!cachedPanoramaUrl && (
            <div className="viewer-placeholder">
              <svg className="viewer-placeholder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <circle cx="12" cy="12" r="10" />
                <path d="M2 12h20" />
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              </svg>
              <span className="viewer-placeholder-text">
                {!taskId ? 'Generate a scene to preview' : 'Generating panorama...'}
              </span>
            </div>
          )}
        </div>

        {/* Segmentation View */}
        <div
          className={`viewer-container segmentation-view ${activeTab === 'segmentation' ? 'active' : 'hidden'}`}
        >
          {loadingAssets ? (
            <div className="viewer-placeholder">
              <div className="spinner-dark spinner"></div>
              <span className="viewer-placeholder-text">Loading segmented assets...</span>
            </div>
          ) : segmentedAssets.length > 0 ? (
            <div className="segmentation-grid">
              {segmentedAssets.map((asset) => (
                <div key={asset.name} className="segmentation-item">
                  <img
                    src={cachedAssetUrls[asset.name] || asset.url}
                    alt={asset.name}
                    onError={(e) => {
                      e.currentTarget.style.display = 'none';
                    }}
                  />
                  <span className="segmentation-item-label">
                    {asset.name.replace(/_/g, ' ')}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="viewer-placeholder">
              <svg className="viewer-placeholder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="3" y="3" width="7" height="7" rx="1" />
                <rect x="14" y="3" width="7" height="7" rx="1" />
                <rect x="3" y="14" width="7" height="7" rx="1" />
                <rect x="14" y="14" width="7" height="7" rx="1" />
              </svg>
              <span className="viewer-placeholder-text">
                {panoramaPath ? 'Analyzing scene for objects...' : 'Waiting for panorama generation...'}
              </span>
            </div>
          )}
        </div>

        {/* Inpainted Viewer */}
        <div
          className={`viewer-container ${activeTab === 'inpainted' ? 'active' : 'hidden'}`}
        >
          <div ref={inpaintedViewerRef} style={{ width: '100%', height: '100%' }} />
          {!cachedInpaintedUrl && (
            <div className="viewer-placeholder">
              <svg className="viewer-placeholder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <circle cx="12" cy="12" r="10" />
                <path d="M2 12h20" />
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              </svg>
              <span className="viewer-placeholder-text">
                {segmentationPath ? 'Inpainting panorama...' : 'Waiting for segmentation...'}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

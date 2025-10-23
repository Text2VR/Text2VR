import React, { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';

interface ProgressiveResultsProps {
  taskId: string | null;
  panoramaPath?: string;
  segmentationPath?: string;
  inpaintedPath?: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
}

interface SegmentedAsset {
  name: string;
  url: string;
}

export const ProgressiveResults: React.FC<ProgressiveResultsProps> = ({
  taskId,
  panoramaPath,
  segmentationPath,
  inpaintedPath,
  status,
}) => {
  const [segmentedAssets, setSegmentedAssets] = useState<SegmentedAsset[]>([]);
  const [loadingAssets, setLoadingAssets] = useState(false);

  const panoramaUrl = panoramaPath && taskId ? apiService.getPanoramaUrl(taskId) : null;
  const inpaintedUrl = inpaintedPath && taskId ? apiService.getInpaintedUrl(taskId) : null;

  // Fetch segmented assets when segmentation is complete
  useEffect(() => {
    if (taskId && segmentationPath) {
      setLoadingAssets(true);
      apiService.getSegmentationAssets(taskId)
        .then(assets => {
          setSegmentedAssets(assets);
          setLoadingAssets(false);
        })
        .catch(err => {
          console.error('Failed to load segmented assets:', err);
          setLoadingAssets(false);
        });
    }
  }, [taskId, segmentationPath]);

  if (!taskId) return null;

  return (
    <div style={{ marginTop: '30px' }}>
      <h2 style={{
        marginBottom: '25px',
        color: '#333',
        textAlign: 'center',
        fontSize: '24px'
      }}>
        Progressive Results
      </h2>

      {/* Panorama Section */}
      <div style={{
        marginBottom: '30px',
        padding: '20px',
        backgroundColor: '#f9f9f9',
        borderRadius: '10px',
        border: '2px solid #e0e0e0'
      }}>
        <h3 style={{
          marginBottom: '15px',
          color: '#444',
          fontSize: '18px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <span>{panoramaPath ? '✅' : '⏳'}</span>
          <span>1. Generated Panorama</span>
        </h3>
        {panoramaUrl ? (
          <div>
            <img
              src={panoramaUrl}
              alt="Generated panorama"
              style={{
                width: '100%',
                height: 'auto',
                borderRadius: '8px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
              }}
              onError={(e) => {
                console.error('Failed to load panorama image');
                e.currentTarget.style.display = 'none';
              }}
            />
          </div>
        ) : (
          <div style={{
            padding: '40px',
            textAlign: 'center',
            color: '#999',
            fontSize: '14px',
            backgroundColor: '#fff',
            borderRadius: '8px',
            border: '1px dashed #ddd'
          }}>
            {status === 'queued' ? 'Waiting to start...' :
             status === 'processing' ? 'Generating panorama...' :
             status === 'failed' ? 'Generation failed' : 'Waiting...'}
          </div>
        )}
      </div>

      {/* Segmentation Section */}
      <div style={{
        marginBottom: '30px',
        padding: '20px',
        backgroundColor: '#f9f9f9',
        borderRadius: '10px',
        border: '2px solid #e0e0e0'
      }}>
        <h3 style={{
          marginBottom: '15px',
          color: '#444',
          fontSize: '18px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <span>{segmentationPath ? '✅' : '⏳'}</span>
          <span>2. Segmented Objects</span>
        </h3>
        {loadingAssets ? (
          <div style={{
            padding: '40px',
            textAlign: 'center',
            color: '#999',
            fontSize: '14px',
            backgroundColor: '#fff',
            borderRadius: '8px',
            border: '1px dashed #ddd'
          }}>
            Loading segmented objects...
          </div>
        ) : segmentedAssets.length > 0 ? (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
            gap: '15px',
            padding: '10px'
          }}>
            {segmentedAssets.map((asset) => (
              <div key={asset.name} style={{
                backgroundColor: '#fff',
                borderRadius: '8px',
                padding: '10px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                textAlign: 'center'
              }}>
                <img
                  src={asset.url + `?t=${Date.now()}`}
                  alt={asset.name}
                  style={{
                    width: '100%',
                    height: 'auto',
                    borderRadius: '6px',
                    marginBottom: '8px'
                  }}
                  onError={(e) => {
                    console.error(`Failed to load asset: ${asset.name}`);
                    e.currentTarget.style.display = 'none';
                  }}
                />
                <div style={{
                  fontSize: '12px',
                  color: '#666',
                  fontWeight: '500',
                  textTransform: 'capitalize'
                }}>
                  {asset.name.replace(/_/g, ' ')}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{
            padding: '40px',
            textAlign: 'center',
            color: '#999',
            fontSize: '14px',
            backgroundColor: '#fff',
            borderRadius: '8px',
            border: '1px dashed #ddd'
          }}>
            {panoramaPath ? 'Analyzing panorama...' : 'Waiting for panorama...'}
          </div>
        )}
      </div>

      {/* Inpainted Panorama Section */}
      <div style={{
        marginBottom: '30px',
        padding: '20px',
        backgroundColor: '#f9f9f9',
        borderRadius: '10px',
        border: '2px solid #e0e0e0'
      }}>
        <h3 style={{
          marginBottom: '15px',
          color: '#444',
          fontSize: '18px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <span>{inpaintedPath ? '✅' : '⏳'}</span>
          <span>3. Inpainted Panorama (Final Result)</span>
        </h3>
        {inpaintedUrl ? (
          <div>
            <img
              src={inpaintedUrl}
              alt="Inpainted panorama"
              style={{
                width: '100%',
                height: 'auto',
                borderRadius: '8px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
              }}
              onError={(e) => {
                console.error('Failed to load inpainted image');
                e.currentTarget.style.display = 'none';
              }}
            />
            <div style={{
              marginTop: '15px',
              padding: '12px',
              backgroundColor: '#e8f5e9',
              borderRadius: '6px',
              color: '#2e7d32',
              fontSize: '14px',
              textAlign: 'center',
              fontWeight: '500'
            }}>
              Processing complete! This is your final enhanced panorama.
            </div>
          </div>
        ) : (
          <div style={{
            padding: '40px',
            textAlign: 'center',
            color: '#999',
            fontSize: '14px',
            backgroundColor: '#fff',
            borderRadius: '8px',
            border: '1px dashed #ddd'
          }}>
            {segmentationPath ? 'Inpainting panorama...' : 'Waiting for segmentation...'}
          </div>
        )}
      </div>
    </div>
  );
};

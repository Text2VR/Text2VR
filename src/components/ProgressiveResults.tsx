import React from 'react';
import { apiService } from '../services/apiService';

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
  status,
}) => {
  if (!taskId) return null;

  const panoramaUrl = panoramaPath && taskId ? apiService.getPanoramaUrl(taskId) : null;
  const segmentationUrl = segmentationPath && taskId ? apiService.getSegmentationUrl(taskId) : null;
  const inpaintedUrl = inpaintedPath && taskId ? apiService.getInpaintedUrl(taskId) : null;

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
          <span>2. Segmentation Visualization</span>
        </h3>
        {segmentationUrl ? (
          <div>
            <img
              src={segmentationUrl}
              alt="Segmentation visualization"
              style={{
                width: '100%',
                height: 'auto',
                borderRadius: '8px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
              }}
              onError={(e) => {
                console.error('Failed to load segmentation image');
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

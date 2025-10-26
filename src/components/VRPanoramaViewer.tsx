import React, { useEffect, useRef } from 'react';

// Import A-Frame
import 'aframe';

interface VRPanoramaViewerProps {
  taskId: string | null;
  visible: boolean;
  imageUrl?: string;
}

export const VRPanoramaViewer: React.FC<VRPanoramaViewerProps> = ({ 
  taskId, 
  visible, 
  imageUrl 
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (visible && taskId && containerRef.current) {
      const panoramaUrl = imageUrl || `/panorama/${taskId}?t=${Date.now()}`;
      
      // A-Frame 씬을 동적으로 생성
      containerRef.current.innerHTML = `
        <a-scene 
          embedded
          style="width: 100%; height: 100%;"
          vr-mode-ui="enabled: true"
          background="color: #000"
        >
          <a-sky 
            src="${panoramaUrl}"
            rotation="0 -130 0"
          ></a-sky>
          
          <a-entity
            id="cameraRig"
            position="0 1.6 0"
          >
            <a-camera
              look-controls="enabled: true"
              wasd-controls="enabled: false"
              position="0 0 0"
            ></a-camera>
          </a-entity>

          <a-light type="ambient" color="#404040"></a-light>
          <a-light type="directional" position="0 1 1" color="#ffffff"></a-light>
        </a-scene>
      `;
    }
  }, [taskId, visible, imageUrl]);

  if (!visible || !taskId) return null;

  const panoramaUrl = imageUrl || `/panorama/${taskId}?t=${Date.now()}`;

  return (
    <div style={{ marginTop: '30px' }}>
      <h3 style={{ 
        marginBottom: '20px', 
        color: '#333',
        textAlign: 'center' 
      }}>
        Generated VR Panorama:
      </h3>
      
      <div style={{
        marginBottom: '20px',
        textAlign: 'center',
        color: '#666',
        fontSize: '14px'
      }}>
        Click and drag to look around • Use VR headset for immersive experience
      </div>

      <div 
        ref={containerRef}
        style={{
          width: '100%',
          height: '500px',
          border: '2px solid #ddd',
          borderRadius: '10px',
          overflow: 'hidden',
          position: 'relative'
        }}
      />

      {/* 일반 2D 이미지도 함께 제공 */}
      <div style={{ 
        marginTop: '20px',
        textAlign: 'center' 
      }}>
        <h4 style={{ 
          marginBottom: '10px', 
          color: '#666',
          fontSize: '16px'
        }}>
          2D Preview:
        </h4>
        <img
          src={panoramaUrl}
          alt="Generated panorama"
          style={{
            maxWidth: '100%',
            height: 'auto',
            borderRadius: '10px',
            boxShadow: '0 5px 15px rgba(0,0,0,0.1)',
          }}
          onError={(e) => {
            console.error('Failed to load panorama image');
            e.currentTarget.style.display = 'none';
          }}
        />
      </div>
    </div>
  );
};
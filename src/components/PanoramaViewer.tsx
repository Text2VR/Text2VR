import React from 'react';

interface PanoramaViewerProps {
  taskId: string | null;
  visible: boolean;
}

export const PanoramaViewer: React.FC<PanoramaViewerProps> = ({ taskId, visible }) => {
  if (!visible || !taskId) return null;

  return (
    <div
      style={{
        marginTop: '30px',
        textAlign: 'center',
      }}
    >
      <h3 style={{ marginBottom: '20px', color: '#333' }}>Generated Panorama:</h3>
      <img
        src={`/panorama/${taskId}?t=${Date.now()}`}
        alt="Generated panorama"
        style={{
          maxWidth: '100%',
          borderRadius: '10px',
          boxShadow: '0 10px 30px rgba(0,0,0,0.2)',
        }}
        onError={(e) => {
          console.error('Failed to load panorama image');
          e.currentTarget.style.display = 'none';
        }}
        onLoad={() => {
          console.log('Panorama image loaded successfully');
        }}
      />
    </div>
  );
};
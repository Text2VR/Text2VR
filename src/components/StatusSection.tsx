import React from 'react';
import { StatusResponse } from '../types/api';

interface StatusSectionProps {
  status: StatusResponse | null;
  visible: boolean;
}

export const StatusSection: React.FC<StatusSectionProps> = ({ status, visible }) => {
  if (!visible || !status) return null;

  const getStatusColor = (statusType: string) => {
    switch (statusType) {
      case 'queued': return '#ffc107';
      case 'processing': return '#17a2b8';
      case 'completed': return '#28a745';
      case 'failed': return '#dc3545';
      default: return '#6c757d';
    }
  };

  return (
    <div
      style={{
        marginTop: '30px',
        padding: '20px',
        background: '#f8f9fa',
        borderRadius: '10px',
      }}
    >
      <div
        style={{
          fontSize: '16px',
          marginBottom: '10px',
          color: getStatusColor(status.status)
        }}
      >
        {status.message}
      </div>
      {status.progress !== undefined && (
        <div
          style={{
            width: '100%',
            height: '8px',
            backgroundColor: '#e9ecef',
            borderRadius: '4px',
            overflow: 'hidden',
            marginTop: '10px'
          }}
        >
          <div
            style={{
              height: '100%',
              backgroundColor: getStatusColor(status.status),
              width: `${status.progress}%`,
              transition: 'width 0.3s ease'
            }}
          />
        </div>
      )}
    </div>
  );
};
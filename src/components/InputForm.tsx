import React, { useState } from 'react';
import { LoadingSpinner } from './LoadingSpinner';

interface InputFormProps {
  onSubmit: (text: string, sceneName: string) => void;
  isLoading: boolean;
}

export const InputForm: React.FC<InputFormProps> = ({ onSubmit, isLoading }) => {
  const [text, setText] = useState('');
  const [sceneName, setSceneName] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) {
      alert('Please enter a description for your panorama scene.');
      return;
    }
    onSubmit(text.trim(), sceneName.trim());
  };

  return (
    <div style={{ marginBottom: '30px' }}>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '20px' }}>
          <label 
            htmlFor="textInput"
            style={{
              display: 'block',
              marginBottom: '8px',
              fontWeight: 600,
              color: '#555'
            }}
          >
            Describe your panorama scene:
          </label>
          <textarea
            id="textInput"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter a description of the panoramic scene you want to generate (e.g., 'A peaceful forest with tall trees and sunlight filtering through the leaves')"
            disabled={isLoading}
            style={{
              width: '100%',
              padding: '12px',
              border: '2px solid #e0e0e0',
              borderRadius: '10px',
              fontSize: '16px',
              height: '100px',
              resize: 'vertical',
              transition: 'border-color 0.3s ease',
              outline: 'none'
            }}
            onFocus={(e) => {
              e.target.style.borderColor = '#667eea';
              e.target.style.boxShadow = '0 0 0 3px rgba(102, 126, 234, 0.1)';
            }}
            onBlur={(e) => {
              if (!text.trim()) {
                e.target.style.borderColor = '#e0e0e0';
                e.target.style.boxShadow = 'none';
              }
            }}
          />
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label 
            htmlFor="sceneNameInput"
            style={{
              display: 'block',
              marginBottom: '8px',
              fontWeight: 600,
              color: '#555'
            }}
          >
            Scene name (optional):
          </label>
          <input
            type="text"
            id="sceneNameInput"
            value={sceneName}
            onChange={(e) => setSceneName(e.target.value)}
            placeholder="Leave empty for auto-generated name"
            disabled={isLoading}
            style={{
              width: '100%',
              padding: '12px',
              border: '2px solid #e0e0e0',
              borderRadius: '10px',
              fontSize: '16px',
              transition: 'border-color 0.3s ease',
              outline: 'none'
            }}
            onFocus={(e) => {
              e.target.style.borderColor = '#667eea';
              e.target.style.boxShadow = '0 0 0 3px rgba(102, 126, 234, 0.1)';
            }}
            onBlur={(e) => {
              if (!sceneName.trim()) {
                e.target.style.borderColor = '#e0e0e0';
                e.target.style.boxShadow = 'none';
              }
            }}
          />
        </div>

        <button
          type="submit"
          disabled={isLoading}
          style={{
            width: '100%',
            padding: '15px',
            background: isLoading ? '#ccc' : 'linear-gradient(45deg,rgb(94, 94, 94),rgb(94, 94, 94))',
            color: 'white',
            border: 'none',
            borderRadius: '10px',
            fontSize: '18px',
            fontWeight: 600,
            cursor: isLoading ? 'not-allowed' : 'pointer',
            transition: 'transform 0.2s ease',
            transform: isLoading ? 'none' : 'translateY(0)',
            opacity: isLoading ? 0.6 : 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px'
          }}
          onMouseEnter={(e) => {
            if (!isLoading) {
              e.currentTarget.style.transform = 'translateY(-2px)';
            }
          }}
          onMouseLeave={(e) => {
            if (!isLoading) {
              e.currentTarget.style.transform = 'translateY(0)';
            }
          }}
        >
          {isLoading && <LoadingSpinner />}
          {isLoading ? 'Generating...' : 'Generate Panorama'}
        </button>
      </form>
    </div>
  );
};
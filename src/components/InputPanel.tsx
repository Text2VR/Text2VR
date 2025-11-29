import React, { useState } from 'react';

interface InputPanelProps {
  onSubmit: (text: string, sceneName: string, options: GenerationOptions) => void;
  isLoading: boolean;
}

export interface GenerationOptions {
  useSelfRefinement: boolean;
  numPrompt: number;
  maxRounds: number;
}

export const InputPanel: React.FC<InputPanelProps> = ({ onSubmit, isLoading }) => {
  const [text, setText] = useState('');
  const [sceneName, setSceneName] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [options, setOptions] = useState<GenerationOptions>({
    useSelfRefinement: false,
    numPrompt: 3,
    maxRounds: 3,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) {
      return;
    }
    onSubmit(text.trim(), sceneName.trim(), options);
  };

  return (
    <div className="input-panel">
      <div className="panel-card">
        <div className="panel-header">
          <svg className="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
          <h2 className="panel-title">Scene Input</h2>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="text-input" className="form-label">
              Describe your scene
            </label>
            <textarea
              id="text-input"
              className="form-textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="A cozy living room with large windows, modern furniture, and warm ambient lighting..."
              disabled={isLoading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="scene-name" className="form-label">
              Scene name (optional)
            </label>
            <input
              type="text"
              id="scene-name"
              className="form-input"
              value={sceneName}
              onChange={(e) => setSceneName(e.target.value)}
              placeholder="my-living-room"
              disabled={isLoading}
            />
          </div>

          <div className="form-group">
            <button
              type="button"
              className={`advanced-toggle ${showAdvanced ? 'open' : ''}`}
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <svg className="advanced-toggle-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
              Advanced Options
            </button>

            {showAdvanced && (
              <div className="advanced-options">
                <div className="option-row">
                  <span className="option-label">Self Refinement</span>
                  <div
                    className={`option-toggle ${options.useSelfRefinement ? 'active' : ''}`}
                    onClick={() => setOptions({ ...options, useSelfRefinement: !options.useSelfRefinement })}
                  />
                </div>

                <div className="option-row">
                  <span className="option-label">Number of Prompts</span>
                  <input
                    type="number"
                    className="option-input"
                    min="1"
                    max="10"
                    value={options.numPrompt}
                    onChange={(e) => setOptions({ ...options, numPrompt: parseInt(e.target.value) || 3 })}
                    disabled={isLoading}
                  />
                </div>

                <div className="option-row">
                  <span className="option-label">Max Rounds</span>
                  <input
                    type="number"
                    className="option-input"
                    min="1"
                    max="5"
                    value={options.maxRounds}
                    onChange={(e) => setOptions({ ...options, maxRounds: parseInt(e.target.value) || 3 })}
                    disabled={isLoading}
                  />
                </div>
              </div>
            )}
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={isLoading || !text.trim()}
          >
            {isLoading ? (
              <>
                <span className="spinner" />
                Generating...
              </>
            ) : (
              <>
                <svg className="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
                Generate Scene
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
};

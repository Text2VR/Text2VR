import React, { useState, useEffect, useRef } from 'react';
import { Header } from './components/Header';
import { PipelineStepper, PipelineStage, parseStageFromMessage } from './components/PipelineStepper';
import { InputPanel, GenerationOptions } from './components/InputPanel';
import { ResultPanel } from './components/ResultPanel';
import { DownloadHub } from './components/DownloadHub';
import { apiService } from './services/apiService';
import { StatusResponse } from './types/api';
import './App.css';

export const App: React.FC = () => {
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentStage, setCurrentStage] = useState<PipelineStage>('idle');

  // Result paths
  const [panoramaPath, setPanoramaPath] = useState<string | null>(null);
  const [segmentationPath, setSegmentationPath] = useState<string | null>(null);
  const [inpaintedPath, setInpaintedPath] = useState<string | null>(null);
  const [asset3dPaths, setAsset3dPaths] = useState<Record<string, string> | null>(null);
  const [plyPath, setPlyPath] = useState<string | null>(null);
  const [sceneName, setSceneName] = useState<string | null>(null);

  const statusCheckInterval = useRef<NodeJS.Timeout | null>(null);

  const startStatusChecking = (taskId: string) => {
    if (statusCheckInterval.current) {
      clearInterval(statusCheckInterval.current);
    }

    statusCheckInterval.current = setInterval(async () => {
      try {
        const statusResponse = await apiService.getStatus(taskId);
        setStatus(statusResponse);

        // Update pipeline stage based on message
        const stage = parseStageFromMessage(statusResponse.message);
        if (stage !== 'idle') {
          setCurrentStage(stage);
        }

        // Update progressive results paths
        if (statusResponse.panorama_path) {
          setPanoramaPath(statusResponse.panorama_path);
        }
        if (statusResponse.segmentation_visualization_path) {
          setSegmentationPath(statusResponse.segmentation_visualization_path);
        }
        if (statusResponse.inpainted_panorama_path) {
          setInpaintedPath(statusResponse.inpainted_panorama_path);
        }
        if (statusResponse.asset_3d_paths) {
          setAsset3dPaths(statusResponse.asset_3d_paths);
        }
        if (statusResponse.ply_path) {
          setPlyPath(statusResponse.ply_path);
        }
        if (statusResponse.scene_name) {
          setSceneName(statusResponse.scene_name);
        }

        if (statusResponse.status === 'completed') {
          setCurrentStage('completed');
          stopStatusChecking();
          setIsLoading(false);
        } else if (statusResponse.status === 'failed') {
          setCurrentStage('failed');
          stopStatusChecking();
          setIsLoading(false);
        }
      } catch (error) {
        console.error('Status check error:', error);
        stopStatusChecking();
        setStatus({
          status: 'failed',
          message: `Status check error: ${error instanceof Error ? error.message : 'Unknown error'}`,
          task_id: taskId
        });
        setCurrentStage('failed');
        setIsLoading(false);
      }
    }, 2000);
  };

  const stopStatusChecking = () => {
    if (statusCheckInterval.current) {
      clearInterval(statusCheckInterval.current);
      statusCheckInterval.current = null;
    }
  };

  const handleGenerate = async (text: string, inputSceneName: string, options: GenerationOptions) => {
    setIsLoading(true);
    setStatus(null);
    setCurrentStage('query_rewrite');

    // Reset all paths
    setPanoramaPath(null);
    setSegmentationPath(null);
    setInpaintedPath(null);
    setAsset3dPaths(null);
    setPlyPath(null);
    setSceneName(null);

    try {
      const response = await apiService.generatePanorama({
        text,
        scene_name: inputSceneName || null,
        use_self_refinement: options.useSelfRefinement,
        num_prompt: options.numPrompt,
        max_rounds: options.maxRounds,
      });

      setCurrentTaskId(response.task_id);
      setStatus({
        status: 'queued',
        message: response.message,
        task_id: response.task_id
      });

      startStatusChecking(response.task_id);
    } catch (error) {
      console.error('Generation error:', error);
      setStatus({
        status: 'failed',
        message: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        task_id: ''
      });
      setCurrentStage('failed');
      setIsLoading(false);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStatusChecking();
    };
  }, []);

  return (
    <div className="app">
      <Header />

      <PipelineStepper
        currentStage={currentStage}
        status={status?.status || null}
      />

      <main className="main-content">
        <InputPanel
          onSubmit={handleGenerate}
          isLoading={isLoading}
        />

        <div className="result-panel">
          <ResultPanel
            taskId={currentTaskId}
            panoramaPath={panoramaPath || undefined}
            segmentationPath={segmentationPath || undefined}
            inpaintedPath={inpaintedPath || undefined}
            status={status?.status || null}
          />

          <DownloadHub
            taskId={currentTaskId}
            panoramaReady={!!panoramaPath}
            assetsReady={!!asset3dPaths && Object.keys(asset3dPaths).length > 0}
            plyReady={!!plyPath}
            sceneName={sceneName || undefined}
          />

          {status && (
            <div className="status-bar">
              <span className={`status-indicator ${status.status}`} />
              <span className="status-message">{status.message}</span>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

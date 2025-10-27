import React, { useState, useEffect, useRef } from 'react';
import { InputForm } from './components/InputForm';
import { StatusSection } from './components/StatusSection';
import { ProgressiveResults } from './components/ProgressiveResults';
import { apiService } from './services/apiService';
import { StatusResponse } from './types/api';
import './App.css';

export const App: React.FC = () => {
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showPanorama, setShowPanorama] = useState(false);
  const [panoramaPath, setPanoramaPath] = useState<string | null>(null);
  const [segmentationPath, setSegmentationPath] = useState<string | null>(null);
  const [inpaintedPath, setInpaintedPath] = useState<string | null>(null);
  const statusCheckInterval = useRef<NodeJS.Timeout | null>(null);

  const startStatusChecking = (taskId: string) => {
    if (statusCheckInterval.current) {
      clearInterval(statusCheckInterval.current);
    }

    statusCheckInterval.current = setInterval(async () => {
      try {
        const statusResponse = await apiService.getStatus(taskId);
        setStatus(statusResponse);

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

        if (statusResponse.status === 'completed') {
          stopStatusChecking();
          setShowPanorama(true);
          setIsLoading(false);
        } else if (statusResponse.status === 'failed') {
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

  const handleGenerate = async (text: string, sceneName: string) => {
    setIsLoading(true);
    setShowPanorama(false);
    setStatus(null);
    // Reset all paths when starting new generation
    setPanoramaPath(null);
    setSegmentationPath(null);
    setInpaintedPath(null);

    try {
      console.log('[App] Calling generatePanorama with:', { text, sceneName });
      const response = await apiService.generatePanorama({
        text,
        scene_name: sceneName || null
      });

      console.log('[App] Received response:', response);
      console.log('[App] Setting taskId:', response.task_id);

      setCurrentTaskId(response.task_id);
      setStatus({
        status: 'queued',
        message: response.message,
        task_id: response.task_id
      });

      console.log('[App] Starting status checking for:', response.task_id);
      startStatusChecking(response.task_id);
    } catch (error) {
      console.error('[App] Generation error:', error);
      setStatus({
        status: 'failed',
        message: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        task_id: ''
      });
      setIsLoading(false);
    }
  };

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      stopStatusChecking();
    };
  }, []);

  return (
    <div className="app">
      <div className="container" style={{textAlign: "center"}}>
        <img src="../src/logo/logo.png" style={{width:"200px", margin:"0 auto", marginBottom:"30px"}} />
        
        <InputForm 
          onSubmit={handleGenerate}
          isLoading={isLoading}
        />
        
        <StatusSection
          status={status}
          visible={!!status}
        />

        <ProgressiveResults
          taskId={currentTaskId}
          panoramaPath={panoramaPath || undefined}
          segmentationPath={segmentationPath || undefined}
          inpaintedPath={inpaintedPath || undefined}
          status={status?.status || 'queued'}
        />
      </div>
    </div>
  );
};
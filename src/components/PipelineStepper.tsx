import React from 'react';

export type PipelineStage =
  | 'idle'
  | 'query_rewrite'
  | 'panorama_generation'
  | 'segmentation'
  | 'asset_cropping'
  | 'trellis_3d'
  | 'inpainting'
  | 'ply_generation'
  | 'completed'
  | 'failed';

interface PipelineStep {
  id: PipelineStage;
  label: string;
  shortLabel: string;
}

const PIPELINE_STEPS: PipelineStep[] = [
  { id: 'query_rewrite', label: 'Query Rewrite', shortLabel: 'Query' },
  { id: 'panorama_generation', label: 'Panorama', shortLabel: 'Pano' },
  { id: 'segmentation', label: 'Segmentation', shortLabel: 'Seg' },
  { id: 'asset_cropping', label: 'Cropping', shortLabel: 'Crop' },
  { id: 'trellis_3d', label: '3D Generation', shortLabel: '3D' },
  { id: 'inpainting', label: 'Inpainting', shortLabel: 'Inpaint' },
  { id: 'ply_generation', label: 'PLY Export', shortLabel: 'PLY' },
];

interface PipelineStepperProps {
  currentStage: PipelineStage;
  status: 'queued' | 'processing' | 'completed' | 'failed' | null;
}

export const PipelineStepper: React.FC<PipelineStepperProps> = ({
  currentStage,
  status
}) => {
  const getStepStatus = (stepId: PipelineStage, index: number): 'completed' | 'active' | 'pending' => {
    if (status === 'completed' || currentStage === 'completed') {
      return 'completed';
    }

    if (status === 'failed' || currentStage === 'failed') {
      const currentIndex = PIPELINE_STEPS.findIndex(s => s.id === currentStage);
      if (index < currentIndex) return 'completed';
      if (index === currentIndex) return 'active';
      return 'pending';
    }

    const currentIndex = PIPELINE_STEPS.findIndex(s => s.id === currentStage);

    if (currentIndex === -1) {
      return 'pending';
    }

    if (index < currentIndex) return 'completed';
    if (index === currentIndex) return 'active';
    return 'pending';
  };

  if (!status) {
    return null;
  }

  return (
    <div className="pipeline-container">
      <div className="pipeline-stepper">
        {PIPELINE_STEPS.map((step, index) => {
          const stepStatus = getStepStatus(step.id, index);
          const isLast = index === PIPELINE_STEPS.length - 1;

          return (
            <React.Fragment key={step.id}>
              <div className={`pipeline-step ${stepStatus}`}>
                <div className="pipeline-step-indicator">
                  {stepStatus === 'completed' ? (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                  ) : (
                    index + 1
                  )}
                </div>
                <span className="pipeline-step-label">{step.label}</span>
              </div>
              {!isLast && (
                <div className={`pipeline-connector ${stepStatus === 'completed' ? 'completed' : ''}`} />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};

export const parseStageFromMessage = (message: string): PipelineStage => {
  const lowerMessage = message.toLowerCase();

  if (lowerMessage.includes('rewrite') || lowerMessage.includes('query')) {
    return 'query_rewrite';
  }
  if (lowerMessage.includes('panorama') && !lowerMessage.includes('inpaint')) {
    return 'panorama_generation';
  }
  if (lowerMessage.includes('segment')) {
    return 'segmentation';
  }
  if (lowerMessage.includes('crop')) {
    return 'asset_cropping';
  }
  if (lowerMessage.includes('3d') || lowerMessage.includes('trellis')) {
    return 'trellis_3d';
  }
  if (lowerMessage.includes('inpaint')) {
    return 'inpainting';
  }
  if (lowerMessage.includes('ply') || lowerMessage.includes('gaussian') || lowerMessage.includes('point cloud')) {
    return 'ply_generation';
  }
  if (lowerMessage.includes('complete') || lowerMessage.includes('success')) {
    return 'completed';
  }
  if (lowerMessage.includes('fail') || lowerMessage.includes('error')) {
    return 'failed';
  }

  return 'idle';
};

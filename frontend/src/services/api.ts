import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes timeout for video processing
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

// Types for API responses
export interface WeaponTimelineItem {
  timestamp: number;
  confidence: number;
  class_name: string;
  frame_number: number;
}

export interface ViolenceDetectionResult {
  is_violence: boolean;
  confidence: number;
  components: {
    lstm_score: number;
    behavioral_score: number;
    visual_score: number;
  };
  analysis: {
    persons_detected: number;
    movement_patterns: number;
    proximity_interactions: number;
  };
  risk_level: string;
}

export interface WeaponDetectionResult {
  weapons_detected: boolean;
  processing_info: {
    total_frames: number;
    processed_frames: number;
    processing_time_seconds: number;
    frame_skip: number;
    fps: number;
    duration_seconds: number;
  };
  detection_summary: {
    unique_weapons_found: string[];
    total_weapon_detections: number;
    max_confidence: number;
    avg_confidence: number;
    frames_with_weapons: number;
    weapon_presence_percentage: number;
    threat_level: string;
  };
  timeline: WeaponTimelineItem[];
  frame_analysis: {
    total_frames_analyzed: number;
    avg_weapons_per_frame: number;
  };
  recommendations: string[];
  confidence_threshold_used: number;
}

export interface VideoAnalysisResponse {
  filename: string;
  video_info: {
    duration: number;
    fps: number;
    total_frames: number;
    resolution: string;
  };
  analysis: {
    violence_detection: ViolenceDetectionResult | null;
    weapon_detection: WeaponDetectionResult | null;
  };
  summary: {
    threats_detected: boolean;
    risk_level: string;
    primary_concerns: string[];
    recommendations: string[];
    alert_level: string;
    confidence_score: number;
    analysis_timestamp: string;
  };
}

export interface AnalysisProgress {
  status: 'processing' | 'completed' | 'error';
  progress: number;
  message: string;
}

// API service class
export class ForensicAPI {
  static async analyzeVideo(
    file: File,
    onProgress?: (progress: AnalysisProgress) => void,
    abortController?: AbortController
  ): Promise<VideoAnalysisResponse> {
    const formData = new FormData();
    formData.append('file', file);

    // Start with upload progress
    if (onProgress) {
      onProgress({
        status: 'processing',
        progress: 5,
        message: 'Preparing upload...'
      });
    }

    // Calculate progress timing based on video duration
    const getVideoDuration = (file: File): Promise<number> => {
      return new Promise((resolve) => {
        const video = document.createElement('video');
        video.preload = 'metadata';
        video.onloadedmetadata = () => {
          window.URL.revokeObjectURL(video.src);
          resolve(video.duration);
        };
        video.onerror = () => {
          // Fallback to file size if duration can't be read
          const fileSizeMB = file.size / (1024 * 1024);
          resolve(fileSizeMB * 2); // Rough estimate: 2 seconds per MB
        };
        video.src = window.URL.createObjectURL(file);
      });
    };

    const videoDuration = await getVideoDuration(file);
    
    // Realistic progress timing based on actual model processing time
    // Processing ratio: ~5:1 (89 seconds for 18-second video = 4.94x ratio)
    // This accounts for model initialization, frame extraction, and AI processing overhead
    const processingRatio = 4.9; // Slightly under 5 for better UX
    const minimumProcessingTime = 15; // Minimum 15 seconds even for very short videos
    
    // Calculate expected processing time based on video duration
    const expectedProcessingTime = Math.max(
      videoDuration * processingRatio, 
      minimumProcessingTime
    );
    
    // Use consistent 1-second intervals for smooth progress
    const progressInterval = 1000; // 1 second intervals

    try {
      // Start the API request but don't wait for it yet
      const apiRequestPromise = api.post('/analyze/video', formData, {
        signal: abortController?.signal,
      });

      // Create progress simulation that runs independently
      new Promise<void>((resolve) => {
        if (!onProgress) {
          resolve();
          return;
        }

        let currentProgress = 10;
        
        const progressIntervalId = setInterval(() => {
          if (abortController?.signal.aborted) {
            clearInterval(progressIntervalId);
            resolve();
            return;
          }

          // Calculate increment for realistic video-duration-based processing
          let baseIncrement = Math.random() * 2 + 1; // 1-3% base increment
          let actualIncrement = baseIncrement; // Use base increment directly
          
          currentProgress += actualIncrement;
          
          if (currentProgress >= 90) {
            currentProgress = 90; // Stop at 90% and enter final phase
            clearInterval(progressIntervalId);
            
            // Final phase timing based on remaining 10% of expected processing time
            const finalPhaseTime = expectedProcessingTime * 0.1 * 1000; // 10% of total time
            const finalPhaseSteps = 4; // 4 final steps (93%, 96%, 98%, 100%)
            const finalStepInterval = finalPhaseTime / finalPhaseSteps;
            
            // Extended final phases to match actual processing time
            setTimeout(() => {
              if (!abortController?.signal.aborted && onProgress) {
                onProgress({
                  status: 'processing',
                  progress: 93,
                  message: 'Deep learning inference with robustness validation...'
                });
              }
            }, finalStepInterval);
            
            setTimeout(() => {
              if (!abortController?.signal.aborted && onProgress) {
                onProgress({
                  status: 'processing',
                  progress: 96,
                  message: 'Ensemble model aggregation and uncertainty quantification...'
                });
              }
            }, finalStepInterval * 2);
            
            setTimeout(() => {
              if (!abortController?.signal.aborted && onProgress) {
                onProgress({
                  status: 'processing',
                  progress: 98,
                  message: 'Finalizing threat assessment and generating report...'
                });
              }
            }, finalStepInterval * 3);
            
            setTimeout(() => {
              if (!abortController?.signal.aborted && onProgress) {
                onProgress({
                  status: 'completed',
                  progress: 100,
                  message: 'Analysis complete'
                });
              }
              // IMPORTANT: Only resolve after 100% is reached
              setTimeout(() => resolve(), 500); // Give UI time to show 100%
            }, finalStepInterval * 4);
            return;
          }

          let message = 'Processing...';
          if (currentProgress < 30) message = 'Uploading video...';
          else if (currentProgress < 60) message = 'Analyzing video frames...';
          else if (currentProgress < 85) message = 'Running AI detection models...';
          else message = 'Finalizing detection results...';


          onProgress({
            status: 'processing',
            progress: Math.round(currentProgress),
            message
          });
        }, progressInterval);

        // Clean up on abort
        abortController?.signal.addEventListener('abort', () => {
          clearInterval(progressIntervalId);
          resolve();
        });
      });

      // Wait for the API request to complete
      // Don't wait for artificial progress - show results as soon as backend responds
      const response = await apiRequestPromise;
      
      // Stop any ongoing progress simulation since we have real results
      if (onProgress) {
        onProgress({
          status: 'completed',
          progress: 100,
          message: 'Analysis complete'
        });
      }
      
      return response.data;
    } catch (error) {
      // Handle abort specifically
      if (abortController?.signal.aborted) {
        throw new Error('Analysis was cancelled');
      }
      
      if (onProgress) {
        onProgress({
          status: 'error',
          progress: 0,
          message: 'Analysis failed'
        });
      }
      
      if (axios.isAxiosError(error)) {
        throw new Error(
          error.response?.data?.detail || 
          error.response?.data?.message || 
          'Failed to analyze video'
        );
      }
      throw error;
    }
  }

  // Analyze only for violence detection
  static async analyzeViolence(file: File): Promise<{
    filename: string;
    video_info: any;
    violence_detection: ViolenceDetectionResult;
  }> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await api.post('/analyze/violence', formData);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(
          error.response?.data?.detail || 
          error.response?.data?.message || 
          'Failed to analyze violence'
        );
      }
      throw error;
    }
  }

  // Analyze only for weapon detection
  static async analyzeWeapons(file: File): Promise<{
    filename: string;
    video_info: any;
    weapon_detection: WeaponDetectionResult;
  }> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await api.post('/analyze/weapons', formData);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(
          error.response?.data?.detail || 
          error.response?.data?.message || 
          'Failed to analyze weapons'
        );
      }
      throw error;
    }
  }

  // Check API health
  static async checkHealth(): Promise<{
    status: string;
    timestamp: string;
    models_loaded: boolean;
  }> {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error('Backend service unavailable');
      }
      throw error;
    }
  }

  // Get API documentation URL
  static getDocsUrl(): string {
    return `${API_BASE_URL}/docs`;
  }
}

// Helper function to format threat level for display
export const formatThreatLevel = (level: string): {
  color: string;
  bgColor: string;
  icon: string;
} => {
  switch (level.toLowerCase()) {
    case 'critical':
      return {
        color: 'text-crimson-400',
        bgColor: 'bg-crimson-900/20',
        icon: '🚨'
      };
    case 'high':
      return {
        color: 'text-yellow-400',
        bgColor: 'bg-yellow-900/20',
        icon: '⚠️'
      };
    case 'medium':
      return {
        color: 'text-yellow-500',
        bgColor: 'bg-yellow-900/10',
        icon: '⚡'
      };
    case 'low':
      return {
        color: 'text-matrix-400',
        bgColor: 'bg-matrix-900/20',
        icon: '✅'
      };
    default:
      return {
        color: 'text-void-400',
        bgColor: 'bg-void-700',
        icon: '❓'
      };
  }
};

// Helper to format timestamp
export const formatTimestamp = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

export default ForensicAPI;
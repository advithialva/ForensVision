'use client';

import React, { useState, useEffect } from 'react';
import { Clock, Eye, Loader2, Upload, Square, RotateCcw, ArrowLeft, Pause } from 'lucide-react';
import { motion } from 'framer-motion';

interface ViolenceDetectionResult {
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

interface WeaponConfidenceScore {
  confidence: number;
  confidence_level: string;
  detection_threshold: number;
}

interface WeaponDetectionResult {
  weapons_detected: boolean;
  detected_weapons: string[];
  confidence_scores: { [weapon: string]: WeaponConfidenceScore };
  processing_stats: {
    processing_time_seconds: number;
    video_duration_seconds: number;
    total_frames: number;
    frames_processed: number;
    frame_skip_interval: number;
    average_processing_fps: number;
    device_used: string;
    estimated_vs_actual_time: {
      estimated: number;
      actual: number;
      accuracy_percentage: number;
    };
  };
  analysis_summary: {
    status: string;
    message: string;
    processing_efficiency?: string;
    highest_confidence_detection?: {
      weapon: string;
      confidence: number;
      confidence_level: string;
    };
    total_detections?: number;
  };
}

interface ApiAnalysisResult {
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
    detections_found: boolean;
    violence_detected: boolean;
    weapons_detected: boolean;
    analysis_details: {
      violence?: {
        detected: boolean;
        confidence: number;
        description: string;
      };
      weapons?: {
        detected: boolean;
        weapons_found: string[];
        confidence_scores: { [weapon: string]: WeaponConfidenceScore };
        description: string;
        detailed_results?: string[];
      };
    };
    confidence_scores: { [key: string]: number };
    processing_stats: {
      violence?: any;
      weapons?: any;
    };
    analysis_timestamp: string;
  };
}

interface AnalysisResultsProps {
  results: ApiAnalysisResult | null;
  isAnalyzing: boolean;
  onBackToUpload?: () => void;
  onNewAnalysis?: () => void;
  onStopAnalysis?: () => void;
  onRetryAnalysis?: () => void;
  isAnalysisStopped?: boolean;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ 
  results, 
  isAnalyzing, 
  onBackToUpload, 
  onNewAnalysis, 
  onStopAnalysis,
  onRetryAnalysis,
  isAnalysisStopped = false
}) => {
  const [isStopped, setIsStopped] = useState(false);

  // Helper function to get confidence level styling
  const getConfidenceStyle = (confidenceLevel: string) => {
    switch (confidenceLevel) {
      case 'Very High':
        return 'bg-green-500/20 text-green-300 border border-green-400/30';
      case 'High':
        return 'bg-yellow-500/20 text-yellow-300 border border-yellow-400/30';
      case 'Medium':
        return 'bg-orange-500/20 text-orange-300 border border-orange-400/30';
      case 'Low':
        return 'bg-red-500/20 text-red-300 border border-red-400/30';
      case 'Very Low':
        return 'bg-gray-500/20 text-gray-300 border border-gray-400/30';
      default:
        return 'bg-slate-500/20 text-slate-300 border border-slate-400/30';
    }
  };

  // Sync with parent stopped state
  useEffect(() => {
    setIsStopped(isAnalysisStopped);
    
    // Reset stopped state when results come in
    if (results) {
      setIsStopped(false);
    }
  }, [isAnalysisStopped, results]);

  const handleStopAnalysis = () => {
    setIsStopped(true);
    if (onStopAnalysis) {
      onStopAnalysis();
    }
  };

  // Show results immediately if they exist, regardless of analyzing state
  if (!results && (isAnalyzing || isStopped)) {
    return (
      <div className="w-full max-w-6xl mx-auto space-y-6 pb-8">
        <div className="border border-slate-600 bg-slate-800/50 rounded-lg p-8 text-center relative overflow-hidden">
          <div className="space-y-6 relative z-10">
            <div className="flex justify-center">
              {isStopped ? (
                <div className="relative">
                  <Pause className="h-16 w-16 text-slate-400" />
                </div>
              ) : (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="relative"
                >
                  <Loader2 className="h-16 w-16 text-blue-400" />
                </motion.div>
              )}
            </div>
            
            <div className="space-y-3">
              <h3 className="text-3xl font-semibold text-slate-100">
                {isStopped ? "Analysis Paused" : "Analysis in Progress"}
              </h3>
              <p className="text-slate-300 text-lg">
                {isStopped ? "Video analysis has been paused" : "Processing video evidence and running detection algorithms"}
              </p>
              
              <div className="flex justify-center space-x-6 mt-6">
                <div className="text-center">
                  <div className="w-3 h-3 bg-blue-400 rounded-full mx-auto mb-2" />
                  <p className="text-xs text-blue-400 font-mono">Weapon Detection</p>
                </div>
                <div className="text-center">
                  <div className="w-3 h-3 bg-blue-500 rounded-full mx-auto mb-2" />
                  <p className="text-xs text-blue-500 font-mono">Violence Analysis</p>
                </div>
              </div>
            </div>
            
            {/* Estimated Processing Time */}
            <div className="space-y-3 mt-8">
              <div className="text-center">
                <div className="text-sm text-slate-300 mb-2">Estimated Processing Time</div>
                <div className="text-2xl font-bold text-blue-400">15-20 seconds</div>
                <div className="text-xs text-slate-400 mt-1">
                  Processing every 8th frame for optimal speed
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="mt-6 flex justify-center">
              {!isStopped ? (
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleStopAnalysis}
                  className="flex items-center justify-center space-x-2 px-6 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors"
                >
                  <Square className="w-4 h-4" />
                  <span>Stop Analysis</span>
                </motion.button>
              ) : (
                <div className="flex flex-col sm:flex-row gap-4">
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={onRetryAnalysis}
                    className="flex items-center justify-center space-x-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
                  >
                    <RotateCcw className="w-5 h-5" />
                    <span>Retry Analysis</span>
                  </motion.button>
                  
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={onBackToUpload}
                    className="flex items-center justify-center space-x-2 px-6 py-3 bg-slate-600 hover:bg-slate-700 text-slate-200 rounded-lg font-medium transition-colors border border-slate-500"
                  >
                    <ArrowLeft className="w-5 h-5" />
                    <span>Back to Upload</span>
                  </motion.button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!results) {
    return null;
  }

  // Extract simple detection results
  const violence_detection = results.analysis?.violence_detection;
  const weapon_detection = results.analysis?.weapon_detection;
  
  const violenceDetected = violence_detection?.is_violence || false;
  const weaponsDetected = weapon_detection?.weapons_detected || false;
  const violenceConfidence = violence_detection?.confidence ? (violence_detection.confidence * 100) : 0;
  
  // Updated weapon detection variables for new format
  const detectedWeapons = weapon_detection?.detected_weapons || [];
  const weaponCount = detectedWeapons.length;
  const confidenceScores = weapon_detection?.confidence_scores || {};
  
  const processingTime = weapon_detection?.processing_stats?.processing_time_seconds || 0;
  const frameCount = weapon_detection?.processing_stats?.total_frames || 0;

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6 pb-8">
      {/* Detection Results Overview */}
      <motion.div
        initial={{ opacity: 1, y: 0 }}
        animate={{ opacity: 1, y: 0 }}
        className="border border-slate-600 bg-slate-800/50 rounded-lg p-8"
      >
        <div className="text-center space-y-6">
          <h2 className="text-3xl font-semibold text-slate-100">Analysis Results</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Violence Detection Result */}
            <div className="space-y-4">
              <div className={`p-6 rounded-lg border-2 ${violenceDetected ? 'border-red-400 bg-red-900/20' : 'border-green-400 bg-green-900/20'}`}>
                <div className="text-center space-y-3">
                  <div className="text-4xl">
                    {violenceDetected ? '⚠️' : '✅'}
                  </div>
                  <h3 className={`text-xl font-semibold ${violenceDetected ? 'text-red-400' : 'text-green-400'}`}>
                    Violence Detection
                  </h3>
                  <p className={`text-2xl font-bold ${violenceDetected ? 'text-red-300' : 'text-green-300'}`}>
                    {violenceDetected ? 'DETECTED' : 'SAFE'}
                  </p>
                  {violenceDetected && (
                    <p className="text-sm text-slate-300">
                      Confidence: {violenceConfidence.toFixed(1)}%
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Weapon Detection Result */}
            <div className="space-y-4">
              <div className={`p-6 rounded-lg border-2 ${weaponsDetected ? 'border-red-400 bg-red-900/20' : 'border-green-400 bg-green-900/20'}`}>
                <div className="text-center space-y-3">
                  <div className="text-4xl">
                    {weaponsDetected ? '⚠️' : '✅'}
                  </div>
                  <h3 className={`text-xl font-semibold ${weaponsDetected ? 'text-red-400' : 'text-green-400'}`}>
                    Weapon Detection
                  </h3>
                  <p className={`text-2xl font-bold ${weaponsDetected ? 'text-red-300' : 'text-green-300'}`}>
                    {weaponsDetected ? 'FOUND' : 'NONE'}
                  </p>
                  {weaponsDetected && (
                    <p className="text-sm text-slate-300">
                      {weaponCount} detection{weaponCount !== 1 ? 's' : ''}
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
          
          {/* Processing Info - General */}
          {weapon_detection && (
            <div className="flex justify-center space-x-8 text-slate-400 text-sm">
              <div className="flex items-center space-x-2">
                <Clock className="h-4 w-4" />
                <span>Processed in {processingTime.toFixed(1)}s</span>
              </div>
              <div className="flex items-center space-x-2">
                <Eye className="h-4 w-4" />
                <span>{frameCount.toLocaleString()} frames analyzed</span>
              </div>
            </div>
          )}

          {/* Weapon Types if detected */}
          {weaponsDetected && detectedWeapons.length > 0 && (
            <div className="mt-6 p-4 bg-slate-700/50 rounded-lg">
              <h4 className="text-lg font-semibold text-slate-200 mb-4">Suspected Weapons</h4>
              <div className="space-y-3">
                {detectedWeapons.map((weapon: string, index: number) => {
                  const weaponScore = confidenceScores[weapon];
                  return (
                    <div key={index} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg border border-slate-500/30">
                      <div className="flex items-center space-x-3">
                        <span className="px-3 py-1 bg-red-500/20 text-red-300 rounded-full text-sm border border-red-400/30 font-medium">
                          {weapon.charAt(0).toUpperCase() + weapon.slice(1)}
                        </span>
                        {weaponScore && (
                          <span className={`px-2 py-1 text-xs rounded-full font-medium ${getConfidenceStyle(weaponScore.confidence_level)}`}>
                            {weaponScore.confidence_level}
                          </span>
                        )}
                      </div>
                      {weaponScore && (
                        <div className="text-right">
                          <div className="text-slate-200 font-bold text-lg">
                            {(weaponScore.confidence * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-slate-400">
                            Confidence
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="mt-8 flex justify-center">
            <button
              onClick={onNewAnalysis}
              className="flex items-center justify-center space-x-2 px-8 py-3 bg-slate-700 hover:bg-slate-600 text-slate-200 hover:text-white rounded-lg font-medium transition-colors shadow-lg border border-slate-600 hover:border-slate-500"
            >
              <Upload className="w-5 h-5" />
              <span>Analyze Another Video</span>
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default AnalysisResults;
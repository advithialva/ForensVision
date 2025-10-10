'use client';

import React, { useState, useEffect } from 'react';
import { Shield, AlertCircle, Server, Activity, Eye, Cpu, Upload, History } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Navigation from '@/components/Navigation';
import VideoUpload from '@/components/VideoUpload';
import AnalysisResults from '@/components/AnalysisResults';
import { ForensicAPI, VideoAnalysisResponse, AnalysisProgress } from '@/services/api';

export default function ForensicAnalysis() {
  const [activeTab, setActiveTab] = useState('home');
  const [analysisResults, setAnalysisResults] = useState<VideoAnalysisResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [isAnalysisStopped, setIsAnalysisStopped] = useState(false);
  const [analysisController, setAnalysisController] = useState<AbortController | null>(null);

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkBackendHealth = async () => {
    try {
      await ForensicAPI.checkHealth();
      setBackendStatus('online');
    } catch (error) {
      setBackendStatus('offline');
    }
  };

  const handleStartAnalysis = async (file: File) => {
    if (!file) return;

    setCurrentFile(file);
    setActiveTab('upload'); // Stay on upload tab to show progress
    setIsAnalyzing(true);

    const abortController = new AbortController();
    setAnalysisController(abortController);

    try {
      const response = await ForensicAPI.analyzeVideo(
        file,
        (progressUpdate) => {
          setUploadProgress(progressUpdate.progress);
        },
        abortController
      );

      if (!abortController.signal.aborted) {
        // Display results immediately when backend responds
        setAnalysisResults(response);
        setActiveTab('results');
        setIsAnalyzing(false);
      }
    } catch (error: any) {
      if (error.name === 'AbortError' || abortController.signal.aborted) {
        // Analysis was cancelled
        setActiveTab('upload');
        setUploadProgress(0);
      } else {
        console.error('Analysis failed:', error);
        setError('Analysis failed. Please try again.');
      }
      setIsAnalyzing(false);
    }

    setAnalysisController(null);
  };

  // Add the handleFileUpload function that was missing
  const handleFileUpload = (file: File) => {
    handleStartAnalysis(file);
  };

  const handleBackToUpload = () => {
    setActiveTab('upload');
    setAnalysisResults(null);
    setError(null);
    setIsAnalyzing(false);
    setIsAnalysisStopped(false);
    setCurrentFile(null);
  };

  const handleNewAnalysis = () => {
    setActiveTab('upload');
    setAnalysisResults(null);
    setError(null);
    setIsAnalyzing(false);
    setIsAnalysisStopped(false);
    setCurrentFile(null);
  };

  const handleStopAnalysis = () => {
    // Abort the ongoing request
    if (analysisController) {
      analysisController.abort();
    }
    setIsAnalyzing(false);
    setIsAnalysisStopped(true);
    setAnalysisController(null);
  };

  const handleRetryAnalysis = () => {
    if (currentFile) {
      setIsAnalysisStopped(false);
      handleFileUpload(currentFile);
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'upload':
        return (
          <div className="space-y-8">
            <AnimatePresence mode="wait">
              {!isAnalyzing && (
                <motion.div
                  key="upload-header"
                  initial={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  className="text-center space-y-4"
                >
                  <h2 className="text-4xl font-semibold text-slate-100">Evidence Upload</h2>
                  <p className="text-xl text-slate-300">
                    Upload video evidence for multi-modal violence detection and weapon identification analysis
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
            
            <AnimatePresence mode="wait">
              {(isAnalyzing || isAnalysisStopped) ? (
                <motion.div
                  key="analysis-progress"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.1, ease: "easeOut" }}
                >
                  <AnalysisResults
                    results={null}
                    isAnalyzing={isAnalyzing}
                    isAnalysisStopped={isAnalysisStopped}
                    onBackToUpload={handleBackToUpload}
                    onNewAnalysis={handleNewAnalysis}
                    onStopAnalysis={handleStopAnalysis}
                    onRetryAnalysis={handleRetryAnalysis}
                  />
                </motion.div>
              ) : (
                <motion.div
                  key="upload-form"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.1, ease: "easeOut" }}
                >
                  <VideoUpload
                    onFileSelect={handleFileUpload}
                    isUploading={isAnalyzing}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        );

      case 'results':
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.1 }}
          >
            <AnalysisResults
              results={analysisResults as any}
              isAnalyzing={isAnalyzing}
              onBackToUpload={handleBackToUpload}
              onNewAnalysis={handleNewAnalysis}
              onStopAnalysis={handleStopAnalysis}
              onRetryAnalysis={handleRetryAnalysis}
            />
          </motion.div>
        );

      case 'history':
        return (
          <div className="bg-slate-800/50 border border-slate-600/30 rounded-lg p-8 text-center">
            <h2 className="text-4xl font-semibold text-slate-100 mb-4">Case History</h2>
            <p className="text-slate-300">Analysis history and case database coming soon...</p>
          </div>
        );

      default:
        return (
          <div className="space-y-12">
            <div className="text-center space-y-6 relative">
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-4"
              >
                <div className="flex items-center justify-center space-x-4 mb-6">
                  <Shield className="h-16 w-16 text-blue-400 opacity-80" />
                  <h1 className="text-6xl font-semibold font-sans text-slate-100">
                    ForensVision
                  </h1>
                </div>
                  <h2 className="text-2xl font-medium text-gray-300">
                    Uncover Critical Insights. Accelerate Investigations.
                  </h2>
              </motion.div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-slate-800/50 border border-slate-600/30 rounded-lg p-6 text-center transition-all duration-300 hover:border-slate-500/50"
              >
                <Server className={`h-8 w-8 mx-auto mb-3 ${
                  backendStatus === 'online' ? 'text-green-400' : 
                  backendStatus === 'offline' ? 'text-red-400' : 'text-yellow-400'
                }`} />
                <h3 className="font-medium text-slate-200">Backend Status</h3>
                <p className={`font-mono text-sm ${
                  backendStatus === 'online' ? 'text-green-400' : 
                  backendStatus === 'offline' ? 'text-red-400' : 'text-yellow-400'
                }`}>
                  {backendStatus === 'checking' ? 'Checking...' : 
                   backendStatus === 'online' ? 'Online' : 'Offline'}
                </p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-slate-800/50 border border-slate-600/30 rounded-lg p-6 text-center transition-all duration-300 hover:border-slate-500/50"
              >
                <Eye className="h-8 w-8 mx-auto mb-3 text-blue-400" />
                <h3 className="font-medium text-slate-200">AI Models</h3>
                <p className="font-mono text-sm text-green-400">Loaded</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-slate-800/50 border border-slate-600/30 rounded-lg p-6 text-center transition-all duration-300 hover:border-slate-500/50"
              >
                <Activity className="h-8 w-8 mx-auto mb-3 text-blue-400" />
                <h3 className="font-medium text-slate-200">Analysis Engine</h3>
                <p className="font-mono text-sm text-green-400">Ready</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-slate-800/50 border border-slate-600/30 rounded-lg p-6 text-center transition-all duration-300 hover:border-slate-500/50"
              >
                <Cpu className="h-8 w-8 mx-auto mb-3 text-blue-400" />
                <h3 className="font-medium text-slate-200">Processing</h3>
                <p className="font-mono text-sm text-green-400">8-12 seconds</p>
              </motion.div>
            </div>

            <div className="bg-slate-800/50 border border-slate-600/30 rounded-lg p-8">
              <h3 className="text-2xl font-semibold text-slate-100 mb-6 text-center">Quick Actions</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setActiveTab('upload')}
                  className="bg-slate-700/50 border border-slate-600/30 hover:border-blue-400/50 p-6 rounded-lg text-left transition-colors"
                >
                  <div className="flex items-center space-x-4">
                    <div className="p-3 rounded-lg bg-blue-400/20 transition-colors">
                      <Activity className="h-8 w-8 text-blue-400" />
                    </div>
                    <div>
                      <h4 className="text-lg font-semibold text-slate-100">Start Analysis</h4>
                      <p className="text-slate-300">Upload video evidence for processing</p>
                    </div>
                  </div>
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setActiveTab('history')}
                  className="bg-slate-700/50 border border-slate-600/30 hover:border-blue-400/50 p-6 rounded-lg text-left transition-colors"
                >
                  <div className="flex items-center space-x-4">
                    <div className="p-3 rounded-lg bg-blue-400/20 transition-colors">
                      <History className="h-8 w-8 text-blue-400" />
                    </div>
                    <div>
                      <h4 className="text-lg font-semibold text-slate-100">Case History</h4>
                      <p className="text-slate-300">View previous analysis results and cases</p>
                    </div>
                  </div>
                </motion.button>
              </div>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 relative">
      <Navigation activeTab={activeTab} onTabChange={setActiveTab} />
      
      <div className="ml-80 min-h-screen overflow-y-auto">
        <div className="container mx-auto px-8 py-12 max-w-full">
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -20, scale: 0.95 }}
                className="mb-8 bg-red-900/20 border border-red-500/30 rounded-lg p-6"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <AlertCircle className="h-8 w-8 text-red-400 flex-shrink-0" />
                    <div>
                      <p className="text-red-400 font-semibold text-lg">System Error</p>
                      <p className="text-red-300 font-mono">{error}</p>
                    </div>
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => setError(null)}
                    className="text-red-400 hover:text-red-300 text-2xl font-bold"
                  >
                    ✕
                  </motion.button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderContent()}
          </motion.div>
        </div>
      </div>
    </div>
  );
}
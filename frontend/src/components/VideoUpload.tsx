'use client';

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileVideo, X, AlertTriangle, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface VideoUploadProps {
  onFileSelect: (file: File) => void;
  isUploading?: boolean;
}

const VideoUpload: React.FC<VideoUploadProps> = ({ 
  onFileSelect, 
  isUploading = false
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    setError(null);
    
    if (rejectedFiles.length > 0) {
      setError('Invalid file format - Only video files accepted (MP4, AVI, MOV, WMV, MKV)');
      return;
    }

    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      
      // Check file size (max 500MB)
      if (file.size > 500 * 1024 * 1024) {
        setError('File size exceeded - Maximum 500MB allowed');
        return;
      }

      setSelectedFile(file);
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.wmv', '.mkv']
    },
    multiple: false,
    disabled: isUploading
  });

  const removeFile = () => {
    setSelectedFile(null);
    setError(null);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6">
      {/* Main Drop Zone */}
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          transition-all duration-300 overflow-hidden bg-slate-800/50
          ${isDragActive 
            ? 'border-blue-400 bg-blue-400/10' 
            : 'border-slate-600 hover:border-blue-500/60 hover:bg-blue-500/5'
          }
          ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <div className="space-y-6 relative z-10">
          <div className="flex justify-center">
            <motion.div
              animate={isDragActive ? { scale: 1.1, rotate: 5 } : { scale: 1, rotate: 0 }}
              transition={{ duration: 0.3 }}
              className="relative"
            >
              {isDragActive ? (
                <div className="relative">
                  <FileVideo className="h-20 w-20 text-blue-400" />
                </div>
              ) : (
                <Upload className="h-20 w-20 text-slate-400 hover:text-blue-400 transition-colors duration-300" />
              )}
            </motion.div>
          </div>
          
          <div className="space-y-3">
            <h3 className="text-2xl font-semibold text-blue-400">
              {isDragActive ? 'Drop Files Here' : 'Upload Evidence'}
            </h3>
            <p className="text-slate-300 text-lg">
              {isDragActive 
                ? 'Release to upload video for analysis'
                : 'Drag and drop video files or click to browse'
              }
            </p>
            <div className="space-y-2">
              <p className="text-sm text-slate-400 font-mono">
                Supported: MP4 • AVI • MOV • WMV • MKV
              </p>
              <p className="text-sm text-slate-500 font-mono">
                Maximum file size: 500MB
              </p>
            </div>
          </div>

          {/* Status indicators */}
          <div className="flex justify-center space-x-4 text-slate-500">
            <div className="w-2 h-2 bg-blue-400 rounded-full" />
            <div className="w-2 h-2 bg-blue-500 rounded-full" />
            <div className="w-2 h-2 bg-blue-600 rounded-full" />
          </div>
        </div>

        {/* Upload progress indicator */}
        {isUploading && (
          <div className="absolute inset-0 rounded-lg overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-blue-400/50" />
          </div>
        )}

        {/* Corner decorations */}
        <div className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-slate-600" />
        <div className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-slate-600" />
        <div className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-slate-600" />
        <div className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-slate-600" />
      </div>
    </div>
  );
};

export default VideoUpload;
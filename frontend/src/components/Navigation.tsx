'use client';

import React from 'react';
import { 
  Home, 
  Upload, 
  History, 
  Shield
} from 'lucide-react';
import { motion } from 'framer-motion';

interface NavItem {
  id: string;
  label: string;
  icon: React.ReactNode;
}

interface NavigationProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const Navigation: React.FC<NavigationProps> = ({ activeTab, onTabChange }) => {
  const navItems: NavItem[] = [
    {
      id: 'home',
      label: 'Dashboard',
      icon: <Home className="w-5 h-5" />,
    },
    {
      id: 'upload',
      label: 'Evidence Upload',
      icon: <Upload className="w-5 h-5" />,
    },
    {
      id: 'history',
      label: 'Case History',
      icon: <History className="w-5 h-5" />,
    },
  ];

  return (
    <div
      className="fixed left-0 top-0 h-full w-80 bg-slate-900/95 backdrop-blur-xl border-r border-slate-700 z-50"
    >
      {/* Header */}
      <div className="p-6 border-b border-slate-700">
        <div className="flex items-center space-x-3">
          <Shield className="w-8 h-8 text-blue-400" />
          <div>
            <h1 className="text-xl font-semibold text-slate-100">
              ForensVision
            </h1>
          </div>
        </div>
      </div>

      {/* Navigation Items */}
      <nav className="mt-8 px-3">
        {navItems.map((item) => (
          <motion.button
            key={item.id}
            onClick={() => onTabChange(item.id)}
            className={`
              w-full flex items-center p-3 mb-2 rounded-lg transition-all duration-300 group
              ${activeTab === item.id 
                ? 'bg-blue-500/20 border border-blue-400/50 text-blue-400' 
                : 'hover:bg-slate-700/50 border border-transparent text-slate-400 hover:text-slate-200'
              }
            `}
            whileHover={{ x: 4 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className={`
              ${activeTab === item.id ? 'text-blue-400' : 'text-slate-400 group-hover:text-slate-200'}
              transition-colors duration-300
            `}>
              {item.icon}
            </div>
            
            <span
              className={`
                ml-3 font-medium text-sm
                ${activeTab === item.id ? 'text-blue-400' : 'text-slate-300 group-hover:text-slate-200'}
                transition-colors duration-300
              `}
            >
              {item.label}
            </span>

            {/* Active indicator */}
            {activeTab === item.id && (
              <motion.div
                layoutId="activeIndicator"
                className="absolute right-0 w-1 h-8 bg-blue-400 rounded-l-full"
                initial={false}
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
              />
            )}
          </motion.button>
        ))}
      </nav>
    </div>
  );
};

export default Navigation;
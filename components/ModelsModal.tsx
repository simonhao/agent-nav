import React, { useState } from 'react';
import { X, Type, Image as ImageIcon, Video, Mic, Database, Layers } from 'lucide-react';
import { Category, Tool } from '../types';
import ToolCard from './ToolCard';

interface ModelsModalProps {
  category: Category;
  isOpen: boolean;
  onClose: () => void;
  accentColor: string;
}

type ModelType = 'All' | 'Text' | 'Image' | 'Video' | 'Audio' | 'Embedding';

const ModelsModal: React.FC<ModelsModalProps> = ({ category, isOpen, onClose, accentColor }) => {
  const [activeTab, setActiveTab] = useState<ModelType>('All');

  if (!isOpen) return null;

  const tabs: { id: ModelType; label: string; icon: React.ElementType }[] = [
    { id: 'All', label: 'All Models', icon: Layers },
    { id: 'Text', label: 'LLMs & Text', icon: Type },
    { id: 'Image', label: 'Image Gen', icon: ImageIcon },
    { id: 'Video', label: 'Video', icon: Video },
    { id: 'Audio', label: 'Audio/TTS', icon: Mic },
    { id: 'Embedding', label: 'Embeddings', icon: Database },
  ];

  const filteredTools = activeTab === 'All' 
    ? category.tools 
    : category.tools.filter(t => t.modelType === activeTab || (activeTab === 'Text' && t.modelType === 'Multimodal'));

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Modal Content */}
      <div className="relative w-full max-w-6xl h-[85vh] bg-slate-50 rounded-xl shadow-2xl flex flex-col overflow-hidden border border-slate-200 animate-in fade-in zoom-in-95 duration-200">
        
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 bg-white border-b border-slate-200">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded bg-${accentColor}/10 text-${accentColor}`}>
              <category.icon size={20} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-slate-800">Global Model Leaderboard</h2>
              <p className="text-xs text-slate-500 font-medium">Top 100+ Trending Models across all modalities</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-2 rounded-full hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors"
          >
            <X size={24} />
          </button>
        </div>

        {/* Tabs */}
        <div className="px-6 pt-4 bg-white/50 border-b border-slate-200 overflow-x-auto">
          <div className="flex gap-6 min-w-max">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    flex items-center gap-2 pb-3 text-sm font-semibold border-b-2 transition-colors
                    ${isActive 
                      ? `border-${accentColor} text-${accentColor}` 
                      : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'}
                  `}
                >
                  <Icon size={16} />
                  {tab.label}
                  <span className={`
                    ml-1 px-1.5 py-0.5 rounded-full text-[10px] 
                    ${isActive ? `bg-${accentColor}/10 text-${accentColor}` : 'bg-slate-100 text-slate-400'}
                  `}>
                    {tab.id === 'All' 
                      ? category.tools.length 
                      : category.tools.filter(t => t.modelType === tab.id || (tab.id === 'Text' && t.modelType === 'Multimodal')).length
                    }
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Grid Content */}
        <div className="flex-1 overflow-y-auto p-6 bg-slate-50/50">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {filteredTools.map((tool) => (
              <ToolCard 
                key={tool.id} 
                tool={tool} 
                accentColor={accentColor} 
              />
            ))}
          </div>
          
          {filteredTools.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-slate-400">
              <Database size={48} className="mb-4 opacity-20" />
              <p>No models found for this category.</p>
            </div>
          )}
        </div>
        
        {/* Footer */}
        <div className="px-6 py-3 bg-white border-t border-slate-200 text-xs text-slate-400 flex justify-between">
           <span>Data sourced from Hugging Face Open LLM Leaderboard, LMSYS Chatbot Arena, and community trends.</span>
           <span>Updated: March 2025</span>
        </div>
      </div>
    </div>
  );
};

export default ModelsModal;

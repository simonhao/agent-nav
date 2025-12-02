import React, { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { Layer } from '../types';
import CategorySection from './CategorySection';

interface LayerContainerProps {
  layer: Layer;
}

const LayerContainer: React.FC<LayerContainerProps> = ({ layer }) => {
  const [isOpen, setIsOpen] = useState(true);

  // Determine column count for desktop view based on category count
  // This creates the "Architecture Diagram" feel where sub-domains are columns
  const gridColsClass = layer.categories.length >= 5 
    ? 'xl:grid-cols-5' 
    : 'xl:grid-cols-4';

  return (
    <div className="bg-white/60 backdrop-blur-sm border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-all duration-300 mb-6">
      {/* Layer Header */}
      <div 
        className={`
          bg-gradient-to-r from-slate-50 to-white 
          border-b border-slate-200/60 px-6 py-3
          flex items-center justify-between cursor-pointer select-none
          hover:bg-slate-50/80 transition-colors
        `}
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center gap-3">
          <div className={`w-1 h-6 rounded-full bg-${layer.color}`}></div>
          <div className="flex flex-col md:flex-row md:items-baseline md:gap-3">
            <h2 className="text-lg font-bold text-slate-800 tracking-tight">
              {layer.name}
            </h2>
            <span className="text-slate-400 text-xs hidden md:inline-block font-medium">
              {layer.description}
            </span>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Visual Badge */}
          <div className={`
            hidden md:block px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider
            bg-${layer.color}/5 text-${layer.color} border border-${layer.color}/20
          `}>
            {layer.categories.length} DOMAINS
          </div>
          
          <button 
            className="p-1 rounded-full hover:bg-slate-200/50 text-slate-400 hover:text-slate-600 transition-colors"
            aria-label={isOpen ? "Collapse layer" : "Expand layer"}
          >
            {isOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </button>
        </div>
      </div>

      {/* Content Area */}
      {isOpen && (
        <div className={`
          p-4 md:p-6 
          grid grid-cols-1 md:grid-cols-2 ${gridColsClass}
          gap-4 md:gap-6 lg:gap-8
          border-t border-slate-100
        `}>
          {layer.categories.map((cat) => (
            <CategorySection 
              key={cat.id} 
              category={cat} 
              accentColor={layer.color} 
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default LayerContainer;
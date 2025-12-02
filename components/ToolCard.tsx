import React, { useState } from 'react';
import { ExternalLink } from 'lucide-react';
import { Tool } from '../types';

interface ToolCardProps {
  tool: Tool;
  accentColor: string;
}

const ToolCard: React.FC<ToolCardProps> = ({ tool, accentColor }) => {
  const [imgError, setImgError] = useState(false);

  const getFavicon = (url: string) => {
    try {
      const domain = new URL(url).hostname;
      return `https://www.google.com/s2/favicons?domain=${domain}&sz=64`;
    } catch (e) {
      return '';
    }
  };

  return (
    <a 
      href={tool.url}
      target="_blank"
      rel="noopener noreferrer"
      className={`
        group relative flex items-center gap-2.5 p-2
        bg-white border border-slate-200 rounded-md
        hover:border-${accentColor}/40 hover:bg-slate-50
        transition-all duration-200 ease-in-out
        shadow-[0_1px_2px_rgba(0,0,0,0.02)] hover:shadow-[0_2px_4px_rgba(0,0,0,0.05)]
        hover:-translate-y-0.5
      `}
    >
      {/* Icon */}
      <div className={`
        flex-shrink-0 w-8 h-8 rounded flex items-center justify-center
        bg-slate-50 border border-slate-100 p-1
        group-hover:border-${accentColor}/20 transition-colors
      `}>
        {!imgError ? (
          <img 
            src={getFavicon(tool.url)} 
            alt={tool.name}
            className="w-full h-full object-contain rounded-sm opacity-90 group-hover:opacity-100"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className={`font-bold text-[10px] text-${accentColor}`}>
            {tool.name.substring(0, 2).toUpperCase()}
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-grow min-w-0">
        <div className="flex items-center justify-between gap-1">
          <h3 className="text-slate-800 font-semibold text-xs truncate group-hover:text-${accentColor} leading-snug">
            {tool.name}
          </h3>
        </div>
        
        <p className="text-slate-400 text-[10px] truncate leading-tight mb-1" title={tool.description}>
          {tool.description}
        </p>

        <div className="flex flex-wrap gap-1">
          {tool.tags.slice(0, 2).map((tag) => (
            <span 
              key={tag} 
              className={`
                px-1 py-[1px] rounded-[2px] text-[8px] font-medium tracking-wide
                bg-slate-100 text-slate-500 border border-slate-200
                group-hover:border-${accentColor}/20 group-hover:text-${accentColor} group-hover:bg-${accentColor}/5
              `}
            >
              {tag}
            </span>
          ))}
        </div>
      </div>
      
      {/* Hover Link Icon */}
      <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
         <ExternalLink className={`w-3 h-3 text-${accentColor}/60`} />
      </div>
    </a>
  );
};

export default ToolCard;
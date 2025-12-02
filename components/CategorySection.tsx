import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Maximize2 } from 'lucide-react';
import { Category } from '../types';
import ToolCard from './ToolCard';
import ModelsModal from './ModelsModal';

interface CategorySectionProps {
  category: Category;
  accentColor: string;
}

const CategorySection: React.FC<CategorySectionProps> = ({ category, accentColor }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  const Icon = category.icon;
  const isModelsCategory = category.id === 'cat-models';
  
  // Default limit
  const LIMIT = 6;
  const showExpandButton = category.tools.length > LIMIT;
  const displayedTools = isExpanded && !isModelsCategory ? category.tools : category.tools.slice(0, LIMIT);

  const handleExpandClick = () => {
    if (isModelsCategory) {
      setIsModalOpen(true);
    } else {
      setIsExpanded(!isExpanded);
    }
  };

  return (
    <>
      <div className="flex flex-col h-full bg-slate-50/50 rounded-lg p-3 border border-slate-100/50 hover:border-slate-200 transition-colors">
        {/* Category Header */}
        <div className="flex items-center gap-2 mb-3 pb-2 border-b border-slate-200/60">
          <div className={`p-1.5 rounded bg-white border border-slate-200 shadow-sm text-${accentColor}`}>
            <Icon size={14} />
          </div>
          <h3 className="text-slate-700 font-bold text-sm tracking-tight truncate" title={category.name}>
            {category.name}
            <span className="ml-1 text-[10px] text-slate-400 font-normal">
              ({category.tools.length})
            </span>
          </h3>
        </div>

        {/* Tools List (Vertical Stack) */}
        <div className="flex flex-col gap-2.5 flex-grow">
          {displayedTools.map((tool) => (
            <ToolCard 
              key={tool.id} 
              tool={tool} 
              accentColor={accentColor} 
            />
          ))}
        </div>

        {/* Expand/Collapse Button */}
        {showExpandButton && (
          <button
            onClick={handleExpandClick}
            className={`
              mt-3 w-full py-1.5 flex items-center justify-center gap-1.5
              text-[11px] font-medium text-slate-500 hover:text-${accentColor}
              bg-white border border-slate-200 rounded hover:bg-slate-50 hover:border-${accentColor}/30
              transition-all duration-200 group
            `}
          >
            {isModelsCategory ? (
              <>
                View All {category.tools.length} Models <Maximize2 size={12} className="group-hover:scale-110 transition-transform"/>
              </>
            ) : (
              isExpanded ? (
                <>
                  Show Less <ChevronUp size={12} />
                </>
              ) : (
                <>
                  Show {category.tools.length - LIMIT} More <ChevronDown size={12} />
                </>
              )
            )}
          </button>
        )}
      </div>

      {/* Render Modal only if it's the models category */}
      {isModelsCategory && (
        <ModelsModal 
          category={category} 
          isOpen={isModalOpen} 
          onClose={() => setIsModalOpen(false)} 
          accentColor={accentColor}
        />
      )}
    </>
  );
};

export default CategorySection;

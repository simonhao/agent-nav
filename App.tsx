import React, { useState, useMemo } from 'react';
import { Search, Command, Zap, Layers } from 'lucide-react';
import { ECOSYSTEM_DATA } from './data';
import LayerContainer from './components/LayerContainer';
import { Layer, Category, Tool } from './types';

const App: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');

  // Filtering Logic
  const filteredData = useMemo(() => {
    if (!searchQuery.trim()) return ECOSYSTEM_DATA;

    const lowerQuery = searchQuery.toLowerCase();

    return ECOSYSTEM_DATA.map((layer: Layer) => {
      const filteredCategories = layer.categories.map((cat: Category) => {
        const filteredTools = cat.tools.filter((tool: Tool) => {
          return (
            tool.name.toLowerCase().includes(lowerQuery) ||
            tool.description.toLowerCase().includes(lowerQuery) ||
            tool.tags.some(tag => tag.toLowerCase().includes(lowerQuery))
          );
        });

        return { ...cat, tools: filteredTools };
      }).filter(cat => cat.tools.length > 0);

      return { ...layer, categories: filteredCategories };
    }).filter(layer => layer.categories.length > 0);

  }, [searchQuery]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans selection:bg-primary-100 selection:text-primary-900">
      {/* Architectural Grid Background */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.4]" 
        style={{ 
          backgroundImage: `
            linear-gradient(to right, #cbd5e1 1px, transparent 1px),
            linear-gradient(to bottom, #cbd5e1 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px'
        }} 
      />
      {/* Subtle dots overlay */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.3]" 
        style={{ 
          backgroundImage: 'radial-gradient(#94a3b8 1px, transparent 1px)', 
          backgroundSize: '10px 10px' 
        }} 
      />

      {/* Header */}
      <header className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-slate-200 shadow-sm">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-1.5 bg-slate-900 rounded-lg shadow-md shadow-slate-900/20">
              <Layers className="text-white w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-900 tracking-tight leading-none">AI Agent Navigator</h1>
              <p className="text-[10px] text-slate-500 font-semibold tracking-wide uppercase mt-0.5">Architecture View</p>
            </div>
          </div>

          <div className="relative w-full max-w-lg ml-6 hidden md:block">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="h-4 w-4 text-slate-400" />
            </div>
            <input
              type="text"
              className="block w-full pl-10 pr-12 py-2 border border-slate-200 rounded-lg leading-5 bg-slate-50 text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-slate-500/20 focus:border-slate-400 sm:text-sm transition-all shadow-inner"
              placeholder="Filter by tool, tag, or category..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
              <kbd className="inline-flex items-center border border-slate-200 rounded px-2 text-[10px] font-sans font-medium text-slate-400">
                ⌘K
              </kbd>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile Search (visible only on small screens) */}
      <div className="md:hidden px-4 py-3 bg-white border-b border-slate-200 sticky top-16 z-40">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-4 w-4 text-slate-400" />
          </div>
          <input
            type="text"
            className="block w-full pl-10 pr-3 py-2 border border-slate-200 rounded-lg leading-5 bg-slate-50 text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-slate-500/20 focus:border-slate-400 text-sm"
            placeholder="Search..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        {filteredData.length > 0 ? (
          <div className="flex flex-col gap-6">
            {filteredData.map((layer) => (
              <LayerContainer key={layer.id} layer={layer} />
            ))}
          </div>
        ) : (
          <div className="text-center py-20 bg-white/50 rounded-xl border border-dashed border-slate-300">
            <div className="inline-flex justify-center items-center w-16 h-16 rounded-full bg-slate-100 border border-slate-200 mb-4">
              <Search className="w-8 h-8 text-slate-400" />
            </div>
            <h3 className="text-lg font-medium text-slate-700">No matches found</h3>
            <p className="text-slate-500 mt-2">Try adjusting your search query.</p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 py-8 mt-8 bg-white/80 backdrop-blur relative z-10">
        <div className="max-w-[1600px] mx-auto px-4 text-center flex flex-col items-center">
          <p className="text-slate-500 text-sm font-medium">
            AI Agent Implementation Map
          </p>
          <p className="text-slate-400 text-xs mt-1">
            Top 10 Global Products · Application to Infrastructure
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;
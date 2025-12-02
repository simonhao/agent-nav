import { LucideIcon } from "lucide-react";

export interface Tool {
  id: string;
  name: string;
  description: string;
  url: string;
  tags: string[];
  featured?: boolean;
  modelType?: 'Text' | 'Image' | 'Video' | 'Audio' | 'Embedding' | 'Multimodal';
}

export interface Category {
  id: string;
  name: string;
  description?: string;
  icon: LucideIcon;
  tools: Tool[];
}

export interface Layer {
  id: string;
  name: string;
  description: string;
  color: string; // Tailwind border color class fragment (e.g., 'blue-500')
  categories: Category[];
}
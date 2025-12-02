import { 
  MessageSquare, Image as ImageIcon, Video, Mic, Users, 
  Database, Shield, Layers, Box, Terminal, FileText, 
  Zap, BookOpen, Globe, Cpu, Lock, Activity, Brain,
  Cloud, Code
} from 'lucide-react';
import { Layer } from './types';

export const ECOSYSTEM_DATA: Layer[] = [
  {
    id: 'layer-app',
    name: 'Application Layer',
    description: 'Conversational agents, coding builders, creative tools, and community ecosystems.',
    color: 'emerald-600',
    categories: [
      {
        id: 'cat-chat',
        name: 'Dialogue & Assistants',
        icon: MessageSquare,
        tools: [
          { id: 'chatgpt', name: 'ChatGPT', description: 'Market-leading general-purpose conversational AI.', url: 'https://chat.openai.com', tags: ['US', 'o3-mini'] },
          { id: 'deepseek-chat', name: 'DeepSeek', description: 'Trending reasoning model (R1) & coding assistant.', url: 'https://chat.deepseek.com', tags: ['CN', 'R1/V3'] },
          { id: 'claude', name: 'Claude', description: 'Anthropic\'s assistant known for coding & nuance.', url: 'https://claude.ai', tags: ['US', 'Sonnet 3.5'] },
          { id: 'grok', name: 'Grok', description: 'xAI\'s assistant with real-time X (Twitter) access.', url: 'https://grok.x.ai', tags: ['US', 'Real-time'] },
          { id: 'perplexity', name: 'Perplexity', description: 'The leading AI-powered answer engine.', url: 'https://www.perplexity.ai', tags: ['US', 'Search'] },
          { id: 'genspark', name: 'Genspark', description: 'Agentic search engine creating custom sparkpages.', url: 'https://www.genspark.ai', tags: ['US/CN', 'Agentic'] },
          { id: 'notebooklm', name: 'NotebookLM', description: 'Google\'s research assistant with Audio Overviews.', url: 'https://notebooklm.google.com', tags: ['US', 'Podcasts'] },
          { id: 'tongyi', name: 'Tongyi', description: 'Alibaba\'s productivity assistant (Qwen Max).', url: 'https://tongyi.aliyun.com', tags: ['CN', 'Productivity'] },
          { id: 'kimi', name: 'Kimi', description: 'Moonshot AI\'s massive context assistant.', url: 'https://kimi.moonshot.cn', tags: ['CN', 'Long Context'] },
          { id: 'gemini', name: 'Gemini', description: 'Google\'s deep native multimodal ecosystem.', url: 'https://gemini.google.com', tags: ['US', 'Multimodal'] },
          { id: 'yuanbao', name: 'Yuanbao', description: 'Tencent\'s AI assistant with WeChat integration.', url: 'https://yuanbao.tencent.com', tags: ['CN', 'WeChat'] },
          { id: 'doubao', name: 'Doubao', description: 'ByteDance\'s dominant consumer AI app.', url: 'https://www.doubao.com', tags: ['CN', 'Daily Use'] }
        ]
      },
      {
        id: 'cat-coding',
        name: 'Coding & Builders',
        icon: Code,
        tools: [
          { id: 'cursor', name: 'Cursor', description: 'The current king of AI code editors (VSCode fork).', url: 'https://www.cursor.com', tags: ['US', 'Editor'] },
          { id: 'windsurf', name: 'Windsurf', description: 'Codeium\'s agentic IDE with deep context awareness.', url: 'https://codeium.com/windsurf', tags: ['US', 'Flow'] },
          { id: 'bolt', name: 'Bolt.new', description: 'Prompt to full-stack web app in the browser.', url: 'https://bolt.new', tags: ['US', 'Web Container'] },
          { id: 'lovable', name: 'Lovable', description: 'GPT Engineer evolved: Design to production code.', url: 'https://lovable.dev', tags: ['EU', 'Full Stack'] },
          { id: 'v0', name: 'v0', description: 'Vercel\'s generative UI system for React/Tailwind.', url: 'https://v0.dev', tags: ['US', 'UI'] },
          { id: 'copilot', name: 'GitHub Copilot', description: 'The industry standard AI pair programmer.', url: 'https://github.com/features/copilot', tags: ['US', 'Enterprise'] },
          { id: 'cline', name: 'Cline', description: 'Open-source autonomous coding agent for VSCode.', url: 'https://github.com/cline/cline', tags: ['Open Source', 'Agent'] },
          { id: 'replit-agent', name: 'Replit Agent', description: 'Build software from scratch with natural language.', url: 'https://replit.com', tags: ['US', 'Cloud IDE'] },
          { id: 'devin', name: 'Devin', description: 'The first fully autonomous AI software engineer.', url: 'https://www.cognition.ai/devin', tags: ['US', 'Autonomous'] },
          { id: 'aider', name: 'Aider', description: 'Powerful AI pair programming in your terminal.', url: 'https://aider.chat', tags: ['US', 'CLI'] }
        ]
      },
      {
        id: 'cat-image',
        name: 'Image Generation',
        icon: ImageIcon,
        tools: [
          { id: 'midjourney', name: 'Midjourney', description: 'Gold standard for artistic texture and lighting.', url: 'https://www.midjourney.com', tags: ['US', 'v6.1'] },
          { id: 'flux', name: 'FLUX.1', description: 'The reigning open-weights champion by Black Forest.', url: 'https://blackforestlabs.ai', tags: ['EU', 'SOTA'] },
          { id: 'recraft', name: 'Recraft', description: 'Best-in-class for vector art and text rendering.', url: 'https://www.recraft.ai', tags: ['UK', 'Design'] },
          { id: 'ideogram', name: 'Ideogram', description: 'Superior typography and prompt adherence.', url: 'https://ideogram.ai', tags: ['CA', 'Text'] },
          { id: 'dalle', name: 'DALL·E 3', description: 'Simple, conversational image generation.', url: 'https://openai.com/dall-e-3', tags: ['US', 'Easy'] },
          { id: 'liblib', name: 'LiblibAI', description: 'Largest Chinese model sharing & gen platform.', url: 'https://www.liblib.art', tags: ['CN', 'Platform'] },
          { id: 'kling-img', name: 'Kling Image', description: 'Kuaishou\'s high-fidelity image generation.', url: 'https://klingai.kuaishou.com', tags: ['CN', 'Realistic'] },
          { id: 'kolors', name: 'Kolors', description: 'Kuaishou\'s open-source photorealistic model.', url: 'https://kolors.kuaishou.com', tags: ['CN', 'Portrait'] },
          { id: 'leonardo', name: 'Leonardo.ai', description: 'Comprehensive suite for game assets.', url: 'https://leonardo.ai', tags: ['AU', 'Assets'] },
          { id: 'civitai', name: 'Civitai', description: 'Hub for Stable Diffusion LoRAs and models.', url: 'https://civitai.com', tags: ['Global', 'Community'] }
        ]
      },
      {
        id: 'cat-video',
        name: 'Video & Motion',
        icon: Video,
        tools: [
          { id: 'kling', name: 'Kling', description: 'Global leader in AI video generation quality.', url: 'https://klingai.kuaishou.com', tags: ['CN', 'Leader'] },
          { id: 'hailuo', name: 'Hailuo (MiniMax)', description: 'Video-01 model: High motion & prompt adherence.', url: 'https://hailuoai.com', tags: ['CN', 'Fast'] },
          { id: 'hunyuan-video', name: 'Hunyuan Video', description: 'Tencent\'s SOTA open-source video model.', url: 'https://github.com/Tencent/HunyuanVideo', tags: ['CN', 'Open Source'] },
          { id: 'wan', name: 'Wan 2.1', description: 'Alibaba\'s newly released powerful video model.', url: 'https://wan.aliyun.com', tags: ['CN', 'New'] },
          { id: 'sora', name: 'Sora', description: 'OpenAI\'s pioneering video model (limited access).', url: 'https://openai.com/sora', tags: ['US', 'Preview'] },
          { id: 'luma', name: 'Luma Dream Machine', description: 'Photon model offers incredible realism.', url: 'https://lumalabs.ai', tags: ['US', 'Ray 2'] },
          { id: 'runway', name: 'Runway', description: 'Gen-3 Alpha with advanced control tools.', url: 'https://runwayml.com', tags: ['US', 'Control'] },
          { id: 'mochi', name: 'Mochi 1', description: 'Genmo\'s high-quality open video model.', url: 'https://www.genmo.ai', tags: ['US', 'Open'] },
          { id: 'vidu', name: 'Vidu', description: 'Rapid generation with long-duration consistency.', url: 'https://www.vidu.studio', tags: ['CN', 'Fast'] },
          { id: 'ltx', name: 'LTX Video', description: 'Lightricks\' open-source efficient video model.', url: 'https://github.com/Lightricks/LTX-Video', tags: ['IL', 'Efficient'] }
        ]
      },
      {
        id: 'cat-audio',
        name: 'Audio & Music',
        icon: Mic,
        tools: [
          { id: 'elevenlabs', name: 'ElevenLabs', description: 'The benchmark for AI speech synthesis.', url: 'https://elevenlabs.io', tags: ['US', 'SOTA'] },
          { id: 'suno', name: 'Suno', description: 'v4 creates radio-ready full songs.', url: 'https://suno.com', tags: ['US', 'Music'] },
          { id: 'udio', name: 'Udio', description: 'Complex musical compositions and editing.', url: 'https://www.udio.com', tags: ['US', 'Pro'] },
          { id: 'fish-audio', name: 'Fish Audio', description: 'Low-latency end-to-end speech models.', url: 'https://fish.audio', tags: ['CN', 'API'] },
          { id: 'cosyvoice', name: 'CosyVoice', description: 'Alibaba\'s excellent open-source speech model.', url: 'https://github.com/FunAudioLLM/CosyVoice', tags: ['CN', 'Open'] },
          { id: 'f5-tts', name: 'F5-TTS', description: 'Non-autoregressive, zero-shot TTS system.', url: 'https://github.com/SWivid/F5-TTS', tags: ['Global', 'Efficient'] },
          { id: 'gpt-sovits', name: 'GPT-SoVITS', description: 'Top-tier few-shot voice cloning.', url: 'https://github.com/RVC-Boss/GPT-SoVITS', tags: ['Global', 'Community'] },
          { id: 'chattts', name: 'ChatTTS', description: 'Optimized for conversational prosody/laughter.', url: 'https://chattts.com', tags: ['CN', 'Dialogue'] },
          { id: 'kyutai', name: 'Moshi', description: 'Real-time native audio-to-audio model.', url: 'https://moshi.chat', tags: ['FR', 'Real-time'] },
          { id: 'hedra', name: 'Hedra', description: 'Audio-driven video character animation.', url: 'https://www.hedra.com', tags: ['US', 'Lip Sync'] }
        ]
      },
      {
        id: 'cat-community',
        name: 'Community',
        icon: Users,
        tools: [
          { id: 'huggingface', name: 'Hugging Face', description: 'The Github of AI models and datasets.', url: 'https://huggingface.co', tags: ['Global', 'Hub'] },
          { id: 'modelscope', name: 'ModelScope', description: 'China\'s premier model-as-a-service hub.', url: 'https://www.modelscope.cn', tags: ['CN', 'Hub'] },
          { id: 'civitai', name: 'Civitai', description: 'The home of open-source image generation.', url: 'https://civitai.com', tags: ['Global', 'Art'] },
          { id: 'github', name: 'GitHub', description: 'Where all open-source AI code lives.', url: 'https://github.com', tags: ['Global', 'Code'] },
          { id: 'reddit-localllama', name: 'r/LocalLLaMA', description: 'The most active open-weights community.', url: 'https://www.reddit.com/r/LocalLLaMA/', tags: ['Global', 'Forum'] },
          { id: 'discord', name: 'Discord', description: 'Where Alpha testing and communities happen.', url: 'https://discord.com', tags: ['Global', 'Chat'] },
          { id: 'wisemodel', name: 'WiseModel', description: 'Professional Chinese AI community.', url: 'https://www.wisemodel.cn', tags: ['CN', 'Aggregator'] },
          { id: 'producthunt', name: 'Product Hunt', description: 'Daily launches of new AI wrappers/tools.', url: 'https://www.producthunt.com', tags: ['US', 'Launch'] },
          { id: 'kaggle', name: 'Kaggle', description: 'Competitions and high-quality datasets.', url: 'https://www.kaggle.com', tags: ['US', 'Data'] },
          { id: 'x-ai', name: 'X (Twitter)', description: 'Where AI news breaks first.', url: 'https://twitter.com', tags: ['Global', 'News'] }
        ]
      }
    ]
  },
  {
    id: 'layer-dev',
    name: 'Development Layer',
    description: 'Orchestration frameworks, RAG platforms, evaluation, and security.',
    color: 'blue-600',
    categories: [
      {
        id: 'cat-framework',
        name: 'Frameworks',
        icon: Layers,
        tools: [
          { id: 'langchain', name: 'LangChain', description: 'The standard library for LLM applications.', url: 'https://www.langchain.com', tags: ['US', 'Standard'] },
          { id: 'langgraph', name: 'LangGraph', description: 'Build stateful, multi-agent applications.', url: 'https://langchain-ai.github.io/langgraph/', tags: ['US', 'Agents'] },
          { id: 'llamaindex', name: 'LlamaIndex', description: 'The data framework for context-augmented LLMs.', url: 'https://www.llamaindex.ai', tags: ['US', 'Data'] },
          { id: 'pydantic-ai', name: 'PydanticAI', description: 'Type-safe production grade agent framework.', url: 'https://ai.pydantic.dev', tags: ['US', 'Production'] },
          { id: 'crewai', name: 'CrewAI', description: 'Orchestrating role-playing autonomous agents.', url: 'https://www.crewai.com', tags: ['US', 'Roles'] },
          { id: 'autogen', name: 'AutoGen', description: 'Microsoft\'s framework for agentic conversation.', url: 'https://microsoft.github.io/autogen/', tags: ['US', 'MS'] },
          { id: 'semantic-kernel', name: 'Semantic Kernel', description: 'Integrate LLMs with existing code.', url: 'https://github.com/microsoft/semantic-kernel', tags: ['US', 'Enterprise'] },
          { id: 'metagpt', name: 'MetaGPT', description: 'Multi-agent framework capable of software dev.', url: 'https://github.com/geekan/MetaGPT', tags: ['CN', 'SOP'] },
          { id: 'swarm', name: 'Swarm', description: 'OpenAI\'s educational multi-agent framework.', url: 'https://github.com/openai/swarm', tags: ['US', 'Simple'] },
          { id: 'haystack', name: 'Haystack', description: 'Composable NLP for production pipelines.', url: 'https://haystack.deepset.ai', tags: ['EU', 'Pipeline'] }
        ]
      },
      {
        id: 'cat-rag-kb',
        name: 'Knowledge Base & RAG',
        icon: BookOpen,
        tools: [
          { id: 'dify', name: 'Dify.ai', description: 'The ultimate open-source LLM app development platform.', url: 'https://dify.ai', tags: ['CN', 'No-Code'] },
          { id: 'coze', name: 'Coze', description: 'ByteDance\'s powerful bot & plugin platform.', url: 'https://www.coze.com', tags: ['CN', 'Plugins'] },
          { id: 'fastgpt', name: 'FastGPT', description: 'Easy-to-use knowledge base QA system.', url: 'https://fastgpt.in', tags: ['CN', 'KB'] },
          { id: 'ragflow', name: 'RagFlow', description: 'RAG engine with deep document understanding.', url: 'https://ragflow.io', tags: ['CN', 'OCR'] },
          { id: 'langflow', name: 'LangFlow', description: 'Visual flow-builder for RAG apps.', url: 'https://www.langflow.org', tags: ['US', 'Visual'] },
          { id: 'maxkb', name: 'MaxKB', description: 'Out-of-the-box RAG/KB solution.', url: 'https://github.com/1Panel-dev/MaxKB', tags: ['CN', 'Simple'] },
          { id: 'firecrawl', name: 'FireCrawl', description: 'Turn websites into LLM-ready markdown.', url: 'https://www.firecrawl.dev', tags: ['US', 'Scraper'] },
          { id: 'unstructured', name: 'Unstructured', description: 'ETL for LLMs: Ingest any document.', url: 'https://unstructured.io', tags: ['US', 'ETL'] },
          { id: 'anythingllm', name: 'AnythingLLM', description: 'Desktop all-in-one local RAG app.', url: 'https://useanything.com', tags: ['US', 'Local'] },
          { id: 'verba', name: 'Verba', description: 'The Golden RAGtriever by Weaviate.', url: 'https://github.com/weaviate/Verba', tags: ['EU', 'Search'] }
        ]
      },
      {
        id: 'cat-eval-monitor',
        name: 'Evaluation & Monitoring',
        icon: Activity,
        tools: [
          { id: 'langsmith', name: 'LangSmith', description: 'DevOps for LLMs: Debug, Test, Monitor.', url: 'https://smith.langchain.com', tags: ['US', 'DevOps'] },
          { id: 'langfuse', name: 'Langfuse', description: 'Open source LLM engineering platform.', url: 'https://langfuse.com', tags: ['DE', 'Trace'] },
          { id: 'arize', name: 'Arize Phoenix', description: 'AI observability & evaluation.', url: 'https://arize.com/phoenix', tags: ['US', 'Trace'] },
          { id: 'weights', name: 'Weights & Biases', description: 'The gold standard for ML experiment tracking.', url: 'https://wandb.ai', tags: ['US', 'MLOps'] },
          { id: 'helicone', name: 'Helicone', description: 'Open-source LLM proxy and monitoring.', url: 'https://www.helicone.ai', tags: ['US', 'Proxy'] },
          { id: 'ragas', name: 'Ragas', description: 'Metrics to evaluate RAG pipelines.', url: 'https://docs.ragas.io', tags: ['Global', 'RAG Eval'] },
          { id: 'deepeval', name: 'DeepEval', description: 'The open-source evaluation framework.', url: 'https://confident-ai.com', tags: ['US', 'Testing'] },
          { id: 'lunary', name: 'Lunary', description: 'Production monitoring & analytics.', url: 'https://lunary.ai', tags: ['EU', 'Analytics'] },
          { id: 'braintrust', name: 'Braintrust', description: 'Enterprise-grade eval and logging.', url: 'https://www.braintrust.dev', tags: ['US', 'Enterprise'] },
          { id: 'trulens', name: 'TruLens', description: 'Honest evaluation for neural nets.', url: 'https://www.trulens.org', tags: ['US', 'Bias'] }
        ]
      },
      {
        id: 'cat-prompt-context',
        name: 'Prompt & Context',
        icon: Brain,
        tools: [
          { id: 'mem0', name: 'Mem0', description: 'The long-term memory layer for AI.', url: 'https://mem0.ai', tags: ['US', 'Memory'] },
          { id: 'zep', name: 'Zep', description: 'Fast, scalable memory for agents.', url: 'https://www.getzep.com', tags: ['US', 'Context'] },
          { id: 'portkey', name: 'Portkey', description: 'The control panel for AI apps.', url: 'https://portkey.ai', tags: ['US', 'Gateway'] },
          { id: 'promptlayer', name: 'PromptLayer', description: 'The first prompt engineering platform.', url: 'https://promptlayer.com', tags: ['US', 'CMS'] },
          { id: 'agenta', name: 'Agenta', description: 'Collaborative prompt management.', url: 'https://agenta.ai', tags: ['DE', 'Open Source'] },
          { id: 'pezzo', name: 'Pezzo', description: 'Cloud-native LLMOps platform.', url: 'https://github.com/pezzolabs/pezzo', tags: ['Global', 'GraphQL'] },
          { id: 'prompt-perfect', name: 'Prompt Perfect', description: 'Optimize prompts for any model.', url: 'https://promptperfect.jina.ai', tags: ['Global', 'Optimize'] },
          { id: 'humanloop', name: 'HumanLoop', description: 'SDK for prompt engineering.', url: 'https://humanloop.com', tags: ['UK', 'SDK'] }
        ]
      },
      {
        id: 'cat-security',
        name: 'Security',
        icon: Lock,
        tools: [
          { id: 'lakera', name: 'Lakera Guard', description: 'Protect against prompt injection.', url: 'https://www.lakera.ai', tags: ['US', 'API'] },
          { id: 'guardrails', name: 'Guardrails AI', description: 'Input/Output validation framework.', url: 'https://www.guardrailsai.com', tags: ['US', 'Validator'] },
          { id: 'llama-guard', name: 'Llama Guard 3', description: 'Meta\'s safety classification model.', url: 'https://llama.meta.com/llama-guard', tags: ['US', 'Model'] },
          { id: 'azure-safety', name: 'Azure AI Safety', description: 'Microsoft\'s content safety filters.', url: 'https://azure.microsoft.com', tags: ['US', 'Cloud'] },
          { id: 'nemo-guardrails', name: 'NeMo Guardrails', description: 'NVIDIA\'s programmable safety rails.', url: 'https://github.com/NVIDIA/NeMo-Guardrails', tags: ['US', 'Enterprise'] },
          { id: 'pyrit', name: 'PyRIT', description: 'Red teaming automation framework.', url: 'https://github.com/Azure/PyRIT', tags: ['US', 'Red Team'] },
          { id: 'gandalf', name: 'Gandalf', description: 'Learn prompt injection by doing.', url: 'https://gandalf.lakera.ai', tags: ['Edu', 'Training'] },
          { id: 'rebuff', name: 'Rebuff', description: 'Multi-stage prompt injection defense.', url: 'https://github.com/protectai/rebuff', tags: ['US', 'Defense'] }
        ]
      }
    ]
  },
  {
    id: 'layer-infra',
    name: 'Infrastructure Layer',
    description: 'Foundation models, vector engines, inference frameworks, and computing providers.',
    color: 'purple-600',
    categories: [
      {
        id: 'cat-models',
        name: 'Foundation Models',
        icon: Box,
        tools: [
          // TEXT MODELS (LLMs)
          { id: 'deepseek-r1', name: 'DeepSeek-R1', description: 'The open reasoning model that shocked the world.', url: 'https://github.com/deepseek-ai/DeepSeek-V3', tags: ['CN', 'Reasoning'], modelType: 'Text' },
          { id: 'gpt4o', name: 'GPT-4o', description: 'OpenAI\'s flagship multimodal frontier model.', url: 'https://openai.com/gpt-4o', tags: ['US', 'SOTA'], modelType: 'Multimodal' },
          { id: 'claude35', name: 'Claude 3.5 Sonnet', description: 'The developer choice for coding & reasoning.', url: 'https://anthropic.com/claude', tags: ['US', 'Coding'], modelType: 'Multimodal' },
          { id: 'llama31', name: 'Llama 3.1 405B', description: 'Meta\'s open-weights frontier contender.', url: 'https://llama.meta.com', tags: ['US', 'Open Weights'], modelType: 'Text' },
          { id: 'qwen25', name: 'Qwen 2.5', description: 'Alibaba\'s dominant general-purpose open model.', url: 'https://qwenlm.github.io', tags: ['CN', 'Best Overall'], modelType: 'Text' },
          { id: 'gemini15pro', name: 'Gemini 1.5 Pro', description: 'Google\'s 2M context window powerhouse.', url: 'https://deepmind.google/technologies/gemini', tags: ['US', 'Long Context'], modelType: 'Multimodal' },
          { id: 'mistrallarge', name: 'Mistral Large 2', description: 'Europe\'s answer to GPT-4.', url: 'https://mistral.ai', tags: ['EU', 'Efficient'], modelType: 'Text' },
          { id: 'grok2', name: 'Grok-2', description: 'xAI\'s frontier model trained on X data.', url: 'https://x.ai', tags: ['US', 'Uncensored'], modelType: 'Text' },
          { id: 'minimax01', name: 'MiniMax-Text-01', description: 'Leading Chinese MoE for roleplay/story.', url: 'https://www.minimaxi.com', tags: ['CN', 'MoE'], modelType: 'Text' },
          { id: 'yi-lightning', name: 'Yi-Lightning', description: '01.AI\'s ultra-fast & smart model.', url: 'https://01.ai', tags: ['CN', 'Speed'], modelType: 'Text' },
          { id: 'glm4', name: 'GLM-4', description: 'Zhipu AI\'s best-in-class general model.', url: 'https://github.com/THUDM/GLM-4', tags: ['CN', 'General'], modelType: 'Text' },
          { id: 'gemma2', name: 'Gemma 2', description: 'Google\'s open weights heavy hitter.', url: 'https://ai.google.dev/gemma', tags: ['US', 'Open'], modelType: 'Text' },
          { id: 'phi35', name: 'Phi-3.5', description: 'Microsoft\'s small but mighty SLM.', url: 'https://azure.microsoft.com/en-us/products/phi', tags: ['US', 'SLM'], modelType: 'Text' },
          { id: 'commandr', name: 'Command R+', description: 'Cohere\'s RAG-optimized enterprise model.', url: 'https://cohere.com/command', tags: ['CA', 'RAG'], modelType: 'Text' },
          { id: 'nemotron', name: 'Nemotron-4', description: 'NVIDIA\'s synthetic data generator.', url: 'https://build.nvidia.com/nvidia/nemotron-4-340b-reward', tags: ['US', 'Synthetic'], modelType: 'Text' },
          { id: 'deepseek-v3', name: 'DeepSeek-V3', description: 'Efficient MoE with GPT-4 performance.', url: 'https://github.com/deepseek-ai/DeepSeek-V3', tags: ['CN', 'Efficient'], modelType: 'Text' },
          { id: 'qwq', name: 'QwQ-32B', description: 'Alibaba\'s reasoning preview model.', url: 'https://huggingface.co/Qwen/QwQ-32B-Preview', tags: ['CN', 'Reasoning'], modelType: 'Text' },
          { id: 'internlm25', name: 'InternLM 2.5', description: 'Shanghai AI Lab\'s strong academic model.', url: 'https://github.com/InternLM/InternLM', tags: ['CN', 'Academic'], modelType: 'Text' },
          { id: 'baichuan4', name: 'Baichuan 4', description: 'Top-tier Chinese medical/legal expertise.', url: 'https://www.baichuan-ai.com', tags: ['CN', 'Domain'], modelType: 'Text' },
          { id: 'moonshot', name: 'Moonshot-v1', description: 'Kimi\'s underlying long-context model.', url: 'https://platform.moonshot.cn', tags: ['CN', '128k'], modelType: 'Text' },
          { id: 'jamba', name: 'Jamba 1.5', description: 'AI21\'s Mamba-Transformer hybrid.', url: 'https://www.ai21.com/jamba', tags: ['IL', 'Hybrid'], modelType: 'Text' },
          { id: 'dbrx', name: 'DBRX', description: 'Databricks\' open standard for coding.', url: 'https://www.databricks.com/blog/introducing-dbrx', tags: ['US', 'Code'], modelType: 'Text' },
          { id: 'arctic', name: 'Snowflake Arctic', description: 'Enterprise-grade open MoE.', url: 'https://www.snowflake.com/en/data-cloud/arctic/', tags: ['US', 'Enterprise'], modelType: 'Text' },
          { id: 'olmo', name: 'OLMo', description: 'AllenAI\'s fully open scientific model.', url: 'https://allenai.org/olmo', tags: ['US', 'Science'], modelType: 'Text' },
          { id: 'smaug', name: 'Smaug-72B', description: 'Abacus AI\'s fine-tuned leaderboard topper.', url: 'https://huggingface.co/abacusai/Smaug-72B-v0.1', tags: ['US', 'Finetune'], modelType: 'Text' },
          { id: 'hunyuan-large', name: 'Hunyuan-Large', description: 'Tencent\'s massive MoE model.', url: 'https://hunyuan.tencent.com', tags: ['CN', 'Scale'], modelType: 'Text' },
          { id: 'spark', name: 'Spark Desk 4.0', description: 'iFlytek\'s cognitive intelligence model.', url: 'https://xinghuo.xfyun.cn', tags: ['CN', 'Voice'], modelType: 'Text' },
          { id: 'sensechat', name: 'SenseChat 5', description: 'SenseTime\'s general multimodal model.', url: 'https://chat.sensetime.com', tags: ['CN', 'Vision'], modelType: 'Text' },
          { id: 'skywork', name: 'Skywork-MoE', description: 'Kunlun\'s massive MoE open weights.', url: 'https://github.com/SkyworkAI/Skywork-MoE', tags: ['CN', 'MoE'], modelType: 'Text' },
          { id: 'wizardlm', name: 'WizardLM-2', description: 'Microsoft\'s evol-instruct masterpiece.', url: 'https://huggingface.co/microsoft/WizardLM-2-8x22B', tags: ['US', 'Instruct'], modelType: 'Text' },
          { id: 'hermes', name: 'Nous Hermes 3', description: 'Uncensored, steerable Llama 3.1 fine-tune.', url: 'https://nousresearch.com', tags: ['US', 'Steerable'], modelType: 'Text' },
          { id: 'athene', name: 'Athene-70B', description: 'Nexusflow\'s high ranking chat model.', url: 'https://nexusflow.ai', tags: ['US', 'Chat'], modelType: 'Text' },
          { id: 'starling', name: 'Starling-7B', description: 'Berkeley\'s RLHF benchmark winner.', url: 'https://starling.cs.berkeley.edu', tags: ['US', 'RLHF'], modelType: 'Text' },
          { id: 'zephyr', name: 'Zephyr 141B', description: 'Hugging Face\'s alignment experiment.', url: 'https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1', tags: ['US', 'Alignment'], modelType: 'Text' },
          { id: 'c4ai', name: 'C4AI Command R', description: 'Cohere for AI research community.', url: 'https://cohere.for.ai', tags: ['CA', 'Research'], modelType: 'Text' },
          
          // IMAGE MODELS
          { id: 'flux1-pro', name: 'FLUX.1 Pro', description: 'The current state-of-the-art image generator.', url: 'https://blackforestlabs.ai', tags: ['EU', 'SOTA'], modelType: 'Image' },
          { id: 'mj-v6', name: 'Midjourney v6.1', description: 'Unmatched artistic style and aesthetics.', url: 'https://midjourney.com', tags: ['US', 'Art'], modelType: 'Image' },
          { id: 'dalle3', name: 'DALL-E 3', description: 'Best prompt following and text rendering.', url: 'https://openai.com/dall-e-3', tags: ['US', 'Easy'], modelType: 'Image' },
          { id: 'sd35', name: 'Stable Diffusion 3.5', description: 'Stability\'s latest open weights model.', url: 'https://stability.ai', tags: ['UK', 'Open'], modelType: 'Image' },
          { id: 'imagen3', name: 'Imagen 3', description: 'Google\'s photorealistic image model.', url: 'https://deepmind.google/technologies/imagen-3', tags: ['US', 'Photo'], modelType: 'Image' },
          { id: 'ideogram2', name: 'Ideogram 2.0', description: 'Leading typography and graphic design.', url: 'https://ideogram.ai', tags: ['CA', 'Text'], modelType: 'Image' },
          { id: 'auraflow', name: 'Auraflow', description: 'Open source flow-based generation.', url: 'https://github.com/fal-ai-models/auraflow', tags: ['US', 'Fast'], modelType: 'Image' },
          { id: 'kolors-model', name: 'Kolors', description: 'Kuaishou\'s superb photorealistic model.', url: 'https://github.com/Kwai-Kolors/Kolors', tags: ['CN', 'Portrait'], modelType: 'Image' },
          { id: 'hunyuan-dit', name: 'Hunyuan-DiT', description: 'Tencent\'s diffusion transformer with Chinese understanding.', url: 'https://github.com/Tencent/HunyuanDiT', tags: ['CN', 'DiT'], modelType: 'Image' },
          { id: 'playground', name: 'Playground v3', description: 'High aesthetic quality for designers.', url: 'https://playgroundai.com', tags: ['US', 'Design'], modelType: 'Image' },
          { id: 'lumina', name: 'Lumina-Next', description: 'Next-gen resolution independent generation.', url: 'https://github.com/Alpha-VLLM/Lumina-T2X', tags: ['CN', 'Resolution'], modelType: 'Image' },
          { id: 'pixart', name: 'PixArt-Sigma', description: 'Efficient high-res image generation.', url: 'https://pixart-alpha.github.io', tags: ['CN', 'Efficient'], modelType: 'Image' },
          { id: 'sdxl', name: 'SDXL 1.0', description: 'The workhorse of local image generation.', url: 'https://stability.ai', tags: ['UK', 'Legacy'], modelType: 'Image' },
          { id: 'proteus', name: 'Proteus', description: 'Data-driven aesthetic enhancement of SDXL.', url: 'https://huggingface.co/dataautogpt3/ProteusV0.4', tags: ['EU', 'Aesthetic'], modelType: 'Image' },
          { id: 'cogview3', name: 'CogView-3', description: 'Zhipu AI\'s dialogue-guided image gen.', url: 'https://github.com/THUDM/CogView3', tags: ['CN', 'Dialogue'], modelType: 'Image' },

          // VIDEO MODELS
          { id: 'sora-model', name: 'Sora', description: 'OpenAI\'s physics-simulating video model.', url: 'https://openai.com/sora', tags: ['US', 'Physics'], modelType: 'Video' },
          { id: 'kling-v1', name: 'Kling 1.5', description: 'High-definition 1080p video generation.', url: 'https://klingai.kuaishou.com', tags: ['CN', 'HD'], modelType: 'Video' },
          { id: 'gen3', name: 'Gen-3 Alpha', description: 'Runway\'s controllable video production model.', url: 'https://runwayml.com', tags: ['US', 'Control'], modelType: 'Video' },
          { id: 'lumaray', name: 'Luma Ray 2', description: 'Fast, high-quality video generation.', url: 'https://lumalabs.ai', tags: ['US', 'Speed'], modelType: 'Video' },
          { id: 'wan21', name: 'Wan 2.1', description: 'Alibaba\'s new powerful video foundation.', url: 'https://wan.aliyun.com', tags: ['CN', 'New'], modelType: 'Video' },
          { id: 'hunyuan-vid', name: 'Hunyuan Video', description: 'Tencent\'s open-source SOTA video model.', url: 'https://github.com/Tencent/HunyuanVideo', tags: ['CN', 'Open'], modelType: 'Video' },
          { id: 'cogvideox', name: 'CogVideoX', description: 'Zhipu AI\'s open-weights video model.', url: 'https://github.com/THUDM/CogVideo', tags: ['CN', 'Open'], modelType: 'Video' },
          { id: 'mochi1', name: 'Mochi 1', description: 'Genmo\'s highly fluid open video model.', url: 'https://github.com/genmoai/mochi', tags: ['US', 'Fluid'], modelType: 'Video' },
          { id: 'ltx-video', name: 'LTX Video', description: 'Lightricks\' efficient open video model.', url: 'https://github.com/Lightricks/LTX-Video', tags: ['IL', 'Efficient'], modelType: 'Video' },
          { id: 'hailuo-01', name: 'Hailuo Video-01', description: 'MiniMax\'s text-to-video with high adherence.', url: 'https://hailuoai.com', tags: ['CN', 'Prompt'], modelType: 'Video' },
          { id: 'vidu-model', name: 'Vidu', description: 'ShengShu\'s long-duration video model.', url: 'https://www.vidu.studio', tags: ['CN', 'Long'], modelType: 'Video' },
          { id: 'veo', name: 'Veo', description: 'Google\'s 1080p video generation model.', url: 'https://deepmind.google/technologies/veo', tags: ['US', '1080p'], modelType: 'Video' },
          { id: 'stable-video', name: 'SVD XT', description: 'Stability AI\'s image-to-video foundation.', url: 'https://stability.ai', tags: ['UK', 'I2V'], modelType: 'Video' },
          { id: 'animatediff', name: 'AnimateDiff', description: 'Motion adapter for Stable Diffusion.', url: 'https://animatediff.github.io', tags: ['CN', 'Adapter'], modelType: 'Video' },
          { id: 'opensora', name: 'OpenSora', description: 'Open-source reproduction of Sora.', url: 'https://github.com/hpcaitech/Open-Sora', tags: ['CN', 'Open'], modelType: 'Video' },
          
          // AUDIO MODELS
          { id: 'eleven-multilingual', name: 'Eleven Multilingual v2', description: 'Indistinguishable from human speech.', url: 'https://elevenlabs.io', tags: ['US', 'TTS'], modelType: 'Audio' },
          { id: 'suno-v4', name: 'Suno v4', description: 'Full song generation with lyrics.', url: 'https://suno.com', tags: ['US', 'Music'], modelType: 'Audio' },
          { id: 'udio-130', name: 'Udio-130', description: 'High fidelity music generation.', url: 'https://udio.com', tags: ['US', 'Music'], modelType: 'Audio' },
          { id: 'whisper-v3', name: 'Whisper v3', description: 'OpenAI\'s universal speech recognition.', url: 'https://github.com/openai/whisper', tags: ['US', 'ASR'], modelType: 'Audio' },
          { id: 'cosyvoice-model', name: 'CosyVoice', description: 'Alibaba\'s instruction-following TTS.', url: 'https://github.com/FunAudioLLM/CosyVoice', tags: ['CN', 'Instruct'], modelType: 'Audio' },
          { id: 'f5tts', name: 'F5-TTS', description: 'Fast, non-autoregressive TTS.', url: 'https://github.com/SWivid/F5-TTS', tags: ['Global', 'Fast'], modelType: 'Audio' },
          { id: 'fish-speech', name: 'Fish Speech', description: 'End-to-end low latency voice model.', url: 'https://fish.audio', tags: ['CN', 'Latency'], modelType: 'Audio' },
          { id: 'chattts-model', name: 'ChatTTS', description: 'Prosody-rich conversational TTS.', url: 'https://chattts.com', tags: ['CN', 'Chat'], modelType: 'Audio' },
          { id: 'gpt-sovits-model', name: 'GPT-SoVITS', description: 'Best few-shot voice cloning.', url: 'https://github.com/RVC-Boss/GPT-SoVITS', tags: ['CN', 'Clone'], modelType: 'Audio' },
          { id: 'parler', name: 'Parler-TTS', description: 'High-quality open-source text-to-speech.', url: 'https://github.com/huggingface/parler-tts', tags: ['US', 'Open'], modelType: 'Audio' },
          { id: 'stable-audio', name: 'Stable Audio 2', description: 'Music and sound FX generation.', url: 'https://stability.ai', tags: ['UK', 'SFX'], modelType: 'Audio' },
          { id: 'audioldm', name: 'AudioLDM-2', description: 'Text-to-audio/music generation.', url: 'https://audioldm.github.io', tags: ['UK', 'Gen'], modelType: 'Audio' },
          { id: 'seamless', name: 'SeamlessM4T', description: 'Meta\'s massivley multilingual translator.', url: 'https://ai.meta.com/research/seamless-communication', tags: ['US', 'Translate'], modelType: 'Audio' },
          
          // EMBEDDING MODELS
          { id: 'openai-embed', name: 'text-embedding-3', description: 'OpenAI\'s latest efficient embeddings.', url: 'https://platform.openai.com/docs/guides/embeddings', tags: ['US', 'Standard'], modelType: 'Embedding' },
          { id: 'voyage', name: 'Voyage-3', description: 'Specialized high-quality retrieval embeddings.', url: 'https://voyageai.com', tags: ['US', 'Retrieval'], modelType: 'Embedding' },
          { id: 'cohere-embed', name: 'Cohere Embed v3', description: 'Multilingual embeddings with compression.', url: 'https://cohere.com/embed', tags: ['CA', 'Multi'], modelType: 'Embedding' },
          { id: 'bge-m3', name: 'BGE-M3', description: 'BAAI\'s dense, sparse, and colbert embedding.', url: 'https://github.com/FlagOpen/FlagEmbedding', tags: ['CN', 'Hybrid'], modelType: 'Embedding' },
          { id: 'e5-mistral', name: 'E5-Mistral-7B', description: 'Microsoft\'s LLM-based embedding model.', url: 'https://github.com/microsoft/unilm', tags: ['US', 'SOTA'], modelType: 'Embedding' },
          { id: 'nomic', name: 'Nomic Embed', description: 'Long-context open-source embeddings.', url: 'https://home.nomic.ai', tags: ['US', 'Open'], modelType: 'Embedding' },
          { id: 'jina', name: 'Jina Embeddings v3', description: '8k context, high performance, compact.', url: 'https://jina.ai', tags: ['DE', 'Compact'], modelType: 'Embedding' },
          { id: 'gte', name: 'GTE-Qwen2', description: 'Alibaba\'s top-ranking embedding model.', url: 'https://huggingface.co/Alibaba-NLP', tags: ['CN', 'Rank 1'], modelType: 'Embedding' },
          { id: 'uae', name: 'UAE-Large-V1', description: 'AnglE optimized universal embeddings.', url: 'https://github.com/SeanLee97/AnglE', tags: ['CN', 'Universal'], modelType: 'Embedding' },
          { id: 'gecko', name: 'Gecko', description: 'Google\'s compact and versatile embeddings.', url: 'https://ai.google.dev', tags: ['US', 'Distilled'], modelType: 'Embedding' }
        ]
      },
      {
        id: 'cat-vectordb',
        name: 'Vector DB',
        icon: Database,
        tools: [
          { id: 'pinecone', name: 'Pinecone', description: 'Serverless vector database for scale.', url: 'https://www.pinecone.io', tags: ['US', 'Managed'] },
          { id: 'milvus', name: 'Milvus', description: 'Cloud-native open-source vector DB.', url: 'https://milvus.io', tags: ['CN', 'Scale'] },
          { id: 'qdrant', name: 'Qdrant', description: 'High-performance Rust-based engine.', url: 'https://qdrant.tech', tags: ['EU', 'Fast'] },
          { id: 'weaviate', name: 'Weaviate', description: 'AI-native vector database.', url: 'https://weaviate.io', tags: ['EU', 'Hybrid'] },
          { id: 'chroma', name: 'Chroma', description: 'The AI-native open-source embedding DB.', url: 'https://www.trychroma.com', tags: ['US', 'Dev'] },
          { id: 'pgvector', name: 'pgvector', description: 'Vector similarity search for Postgres.', url: 'https://github.com/pgvector/pgvector', tags: ['Global', 'SQL'] },
          { id: 'elasticsearch', name: 'Elasticsearch', description: 'Search & analytics engine with vectors.', url: 'https://www.elastic.co', tags: ['US', 'Enterprise'] },
          { id: 'tencent-vdb', name: 'Tencent VDB', description: 'Fully managed enterprise vector DB.', url: 'https://cloud.tencent.com/product/vdb', tags: ['CN', 'Cloud'] },
          { id: 'mongodb', name: 'MongoDB', description: 'Vector search integrated into NoSQL.', url: 'https://www.mongodb.com', tags: ['US', 'Atlas'] },
          { id: 'lancedb', name: 'LanceDB', description: 'Serverless, columnar vector DB.', url: 'https://lancedb.com', tags: ['US', 'Embedded'] }
        ]
      },
      {
        id: 'cat-inference',
        name: 'Inference',
        icon: Cpu,
        tools: [
          { id: 'vllm', name: 'vLLM', description: 'The gold standard for high-throughput inference.', url: 'https://github.com/vllm-project/vllm', tags: ['US', 'PageAttention'] },
          { id: 'ollama', name: 'Ollama', description: 'Run Llama 3, Mistral, Gemma locally.', url: 'https://ollama.com', tags: ['US', 'Local'] },
          { id: 'llamacpp', name: 'llama.cpp', description: 'Inference on standard hardware (Mac/CPU).', url: 'https://github.com/ggerganov/llama.cpp', tags: ['Global', 'GGUF'] },
          { id: 'sglang', name: 'SGLang', description: 'Fast execution for structured generation.', url: 'https://github.com/sgl-project/sglang', tags: ['US', 'Fast'] },
          { id: 'lmdeploy', name: 'LMDeploy', description: 'Efficient toolkit for LLM deployment.', url: 'https://github.com/InternLM/lmdeploy', tags: ['CN', 'Efficient'] },
          { id: 'tgi', name: 'TGI', description: 'Hugging Face\'s production inference container.', url: 'https://github.com/huggingface/text-generation-inference', tags: ['US', 'HF'] },
          { id: 'tensorrt', name: 'TensorRT-LLM', description: 'NVIDIA\'s optimized inference library.', url: 'https://github.com/NVIDIA/TensorRT-LLM', tags: ['US', 'GPU'] },
          { id: 'mnn', name: 'MNN-LLM', description: 'Alibaba\'s lightweight mobile inference.', url: 'https://github.com/alibaba/MNN', tags: ['CN', 'Mobile'] },
          { id: 'localai', name: 'LocalAI', description: 'OpenAI-compatible local API wrapper.', url: 'https://localai.io', tags: ['EU', 'API'] },
          { id: 'exe', name: 'ExLlamaV2', description: 'Fastest inference for modern GPUs.', url: 'https://github.com/turboderp/exllamav2', tags: ['Global', 'Speed'] }
        ]
      },
      {
        id: 'cat-providers',
        name: 'Model Providers',
        icon: Cloud,
        tools: [
          { id: 'deepseek-api', name: 'DeepSeek API', description: 'Extremely affordable token pricing.', url: 'https://platform.deepseek.com', tags: ['CN', 'Cheap'] },
          { id: 'groq', name: 'Groq', description: 'Instant inference speed via LPUs.', url: 'https://groq.com', tags: ['US', 'Instant'] },
          { id: 'siliconflow', name: 'SiliconFlow', description: 'Top tier inference provider in China.', url: 'https://siliconflow.cn', tags: ['CN', 'Hub'] },
          { id: 'openrouter', name: 'OpenRouter', description: 'Access to all models via one API.', url: 'https://openrouter.ai', tags: ['US', 'Aggregator'] },
          { id: 'together', name: 'Together AI', description: 'Serverless training and inference.', url: 'https://together.ai', tags: ['US', 'Cloud'] },
          { id: 'fireworks', name: 'Fireworks AI', description: 'Fast, efficient, production AI.', url: 'https://fireworks.ai', tags: ['US', 'Dev'] },
          { id: 'cerebras', name: 'Cerebras', description: 'Wafer-scale hardware inference.', url: 'https://cerebras.net', tags: ['US', 'Hardware'] },
          { id: 'volcengine', name: 'Volcengine', description: 'ByteDance\'s enterprise Ark platform.', url: 'https://www.volcengine.com', tags: ['CN', 'Ent'] },
          { id: 'bailian', name: 'Aliyun Bailian', description: 'Alibaba\'s Model Studio.', url: 'https://bailian.console.aliyun.com', tags: ['CN', 'Cloud'] },
          { id: 'replicate', name: 'Replicate', description: 'Run AI with an API.', url: 'https://replicate.com', tags: ['US', 'Deploy'] }
        ]
      },
      {
        id: 'cat-papers',
        name: 'Papers & News',
        icon: FileText,
        tools: [
          { id: 'arxiv', name: 'arXiv', description: 'New AI research drops here first.', url: 'https://arxiv.org/list/cs.AI/recent', tags: ['Global', 'Raw'] },
          { id: 'paperswithcode', name: 'Papers w/ Code', description: 'State of the art benchmarks.', url: 'https://paperswithcode.com', tags: ['Global', 'SOTA'] },
          { id: 'huggingface-papers', name: 'HF Papers', description: 'Curated daily top papers.', url: 'https://huggingface.co/papers', tags: ['Global', 'Curated'] },
          { id: 'baai', name: 'BAAI Hub', description: 'Beijing Academy of AI Research.', url: 'https://hub.baai.ac.cn', tags: ['CN', 'Research'] },
          { id: 'alphasignal', name: 'AlphaSignal', description: 'Top rated technical newsletter.', url: 'https://alphasignal.ai', tags: ['US', 'News'] },
          { id: 'synced', name: 'Synced', description: 'Tech insights from China & Global.', url: 'https://syncedreview.com', tags: ['CN', 'Media'] },
          { id: 'tl-dr', name: 'TL;DR AI', description: 'Daily summary of AI news.', url: 'https://tldr.tech/ai', tags: ['US', 'Brief'] },
          { id: 'bens-bites', name: 'Ben\'s Bites', description: 'Digestible daily AI updates.', url: 'https://bensbites.beehiiv.com', tags: ['US', 'Fun'] }
        ]
      }
    ]
  }
];

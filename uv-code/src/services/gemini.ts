import { GoogleGenerativeAI, GenerativeModel, Part } from '@google/generative-ai';
import dotenv from 'dotenv';
import fs from 'fs/promises';

dotenv.config();

// ─── API Keys in priority order (loaded from environment) ────────────────────
const HARDCODED_KEYS: string[] = [
  process.env.GEMINI_API_KEY,
  process.env.REACT_APP_GEMINI_API_KEY,
].filter(Boolean) as string[];

// ─── Models in fallback order (primary → fallback) ───────────────────────────
const MODELS: string[] = [
  'gemini-3.6-flash',   // gemini 3.6 flash (primary)
  'gemini-3.7-flash',   // gemini 3.7 flash (fallback)
];

let language = process.env.APP_LANGUAGE || 'Python';
let customApiKey: string | null = null;

interface Config {
  apiKey: string;
  language: string;
}

function updateConfig(config: Config) {
  if (!config.apiKey) {
    throw new Error('Gemini API key is required');
  }
  customApiKey = config.apiKey.trim();
  language = config.language || 'Python';
}

/** Returns deduplicated key list: custom key first, then env key, then hardcoded keys */
function getKeyList(): string[] {
  const envKey = process.env.GEMINI_API_KEY;
  const candidates: string[] = [];
  if (customApiKey) candidates.push(customApiKey);
  if (envKey) candidates.push(envKey);
  candidates.push(...HARDCODED_KEYS);

  // Deduplicate while preserving order
  const seen = new Set<string>();
  return candidates.filter(k => k && !seen.has(k) && seen.add(k));
}

// Init from env if present
if (process.env.GEMINI_API_KEY) {
  try {
    updateConfig({
      apiKey: process.env.GEMINI_API_KEY,
      language: process.env.APP_LANGUAGE || 'Python',
    });
  } catch (err) {
    console.error('Error initializing Gemini from env:', err);
  }
}

// ─── Types ────────────────────────────────────────────────────────────────────
interface ProcessedSolution {
  approach: string;
  code: string;
  timeComplexity: string;
  spaceComplexity: string;
}

// ─── Optimised System Prompt ──────────────────────────────────────────────────
function buildSystemPrompt(): string {
  return `You are CrackCoder — a world-class AI expert at competitive programming, Data Structures & Algorithms (DSA), SQL & Database queries, MCQ aptitude, and technical interviews.

TASK: Analyse the question(s) shown in the screenshot(s) and provide a perfect, interview-ready solution.

DETECT QUESTION TYPE automatically:
• DSA / Coding Problem  → full working ${language} code + step-by-step approach + complexity analysis
• SQL Query Question     → clean, complete SQL query + query logic breakdown
• MCQ / Aptitude / Quiz  → identify the correct option (e.g. A, B, C, D) + clear justification
• Theoretical / Logical  → concise explanation + direct answer

RESPONSE FORMAT — return ONLY valid JSON (no markdown fences):
{
  "approach": "Clear, concise 3-5 sentence explanation an interviewee would speak aloud. For SQL explain JOINs/CTEs/filters used. For MCQ state 'Answer: X) ...' with rationale. For DSA explain pattern/algorithm.",
  "code": "For coding/DSA: complete, optimal ${language} code with comments. For SQL: clean, production-ready SQL query. For MCQ: correct option letter and answer. Empty string if not applicable.",
  "timeComplexity": "Big-O time complexity with brief reason (write N/A for SQL or non-coding questions)",
  "spaceComplexity": "Big-O space complexity with brief reason (write N/A for SQL or non-coding questions)"
}

RULES:
- Output ONLY the JSON object, nothing else, no markdown code blocks outside JSON.
- For DSA/Coding: Code must be complete, bug-free, and syntactically correct in ${language}.
- For SQL: Write standard ANSI SQL (or engine-specific if mentioned, e.g. MySQL / PostgreSQL / Oracle). Include necessary JOINs, WHERE, GROUP BY, HAVING, or Window Functions.
- For MCQs: Clearly begin approach with 'Answer: <Option Letter>) ...'.
- Prioritise optimal solutions (best time & space complexity).`;
}

// ─── Fallback logic: key × model ─────────────────────────────────────────────
async function callGeminiWithFallback(parts: Part[]): Promise<string> {
  const keys = getKeyList();
  const errors: string[] = [];

  for (const key of keys) {
    for (const model of MODELS) {
      try {
        console.log(`[Gemini] Trying model=${model} key=...${key.slice(-8)}`);
        const genAI = new GoogleGenerativeAI(key);
        const genModel: GenerativeModel = genAI.getGenerativeModel({ model });

        const result = await genModel.generateContent({
          contents: [{ role: 'user', parts }],
          systemInstruction: {
            role: 'system',
            parts: [{ text: buildSystemPrompt() }],
          },
          generationConfig: {
            temperature: 0.3,
            maxOutputTokens: 3000,
            responseMimeType: 'application/json',
          },
        });

        const text = result.response.text();
        console.log(`[Gemini] SUCCESS model=${model} key=...${key.slice(-8)}`);
        return text;
      } catch (err: any) {
        const msg = `model=${model} key=...${key.slice(-8)}: ${err?.message || err}`;
        console.warn(`[Gemini] FAILED — ${msg}`);
        errors.push(msg);
      }
    }
  }

  throw new Error(`All Gemini API keys and models exhausted.\n${errors.join('\n')}`);
}

// ─── Public API ───────────────────────────────────────────────────────────────
export async function processScreenshots(
  screenshots: { path: string }[]
): Promise<ProcessedSolution> {
  const parts: Part[] = [
    { text: 'Analyse the question(s) in the following screenshot(s) and respond per the system instructions.' },
  ];

  for (const screenshot of screenshots) {
    const base64Image = await fs.readFile(screenshot.path, { encoding: 'base64' });
    parts.push({
      inlineData: {
        mimeType: 'image/png',
        data: base64Image,
      },
    });
  }

  const rawText = await callGeminiWithFallback(parts);

  // Strip any accidental markdown fences
  const cleaned = rawText
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();

  try {
    return JSON.parse(cleaned) as ProcessedSolution;
  } catch {
    return {
      approach: cleaned,
      code: '',
      timeComplexity: 'N/A',
      spaceComplexity: 'N/A',
    };
  }
}

export default {
  processScreenshots,
  updateConfig,
};

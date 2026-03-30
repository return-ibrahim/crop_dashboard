// api/analyze.js
// Vercel serverless function — proxies requests to Google Gemini (free tier)
// FREE TIER: https://aistudio.google.com/apikey  (no credit card required)
// Set GEMINI_API_KEY in your Vercel project → Settings → Environment Variables

const GEMINI_MODEL  = 'gemini-2.0-flash';
const MAX_RETRIES   = 4;          // retry up to 4 times on 429
const BASE_DELAY_MS = 2000;       // start with 2 s, doubles each attempt

// ── Helper: sleep ──────────────────────────────────────────────────────────
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── Helper: call Gemini with automatic retry on 429 ───────────────────────
async function callGeminiWithRetry(apiKey, geminiParts, maxOutputTokens) {
  let lastError;

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    if (attempt > 0) {
      // Exponential backoff: 2s → 4s → 8s → 16s
      // Also respect Retry-After header if Gemini sends one
      const delay = BASE_DELAY_MS * Math.pow(2, attempt - 1);
      console.log(`⏳ Rate limited. Waiting ${delay / 1000}s before retry ${attempt}/${MAX_RETRIES - 1}…`);
      await sleep(delay);
    }

    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${apiKey}`,
      {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: geminiParts }],
          generationConfig: {
            maxOutputTokens: maxOutputTokens || 1000,
            temperature:     0.2,
          },
        }),
      }
    );

    // Success
    if (res.ok) {
      const data = await res.json();
      return { ok: true, data };
    }

    // 429 → retry
    if (res.status === 429) {
      const errBody = await res.json().catch(() => ({}));
      lastError = errBody?.error?.message || 'Rate limit exceeded';
      // Check if Gemini sent a Retry-After header (sometimes it does)
      const retryAfter = res.headers.get('Retry-After');
      if (retryAfter && attempt === 0) {
        const waitMs = parseFloat(retryAfter) * 1000;
        if (waitMs > 0 && waitMs < 30000) await sleep(waitMs);
      }
      continue; // go to next attempt
    }

    // Other HTTP error → don't retry
    const errBody = await res.json().catch(() => ({}));
    return {
      ok:     false,
      status: res.status,
      error:  errBody?.error?.message || `Gemini API error (${res.status})`,
    };
  }

  // All retries exhausted
  return {
    ok:    false,
    status: 429,
    error: `Rate limit exceeded after ${MAX_RETRIES} retries. Free tier allows 15 requests/min — please wait a moment and try again. (${lastError})`,
  };
}

// ── Main handler ───────────────────────────────────────────────────────────
export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST')   return res.status(405).json({ error: 'Method Not Allowed' });

  // ── Validate key ─────────────────────────────────────────────────────────
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({
      error: 'GEMINI_API_KEY is not set. Add it in Vercel → Settings → Environment Variables. Get a free key at https://aistudio.google.com/apikey',
    });
  }

  try {
    // ── Convert Anthropic-style request → Gemini format ───────────────────
    const { messages, max_tokens } = req.body;
    const userMessage = messages?.find(m => m.role === 'user');
    if (!userMessage) {
      return res.status(400).json({ error: 'No user message found in request body' });
    }

    const parts = Array.isArray(userMessage.content) ? userMessage.content : [];

    const geminiParts = parts.map(part => {
      if (part.type === 'text') return { text: part.text };
      if (part.type === 'image') {
        return {
          inline_data: {
            mime_type: part.source?.media_type || 'image/jpeg',
            data:       part.source?.data       || '',
          },
        };
      }
      return null;
    }).filter(Boolean);

    // ── Call Gemini (with retry) ──────────────────────────────────────────
    const result = await callGeminiWithRetry(apiKey, geminiParts, max_tokens);

    if (!result.ok) {
      return res.status(result.status || 500).json({ error: result.error });
    }

    // ── Normalize Gemini response → Anthropic shape ───────────────────────
    const text = result.data.candidates?.[0]?.content?.parts?.[0]?.text || '';
    return res.status(200).json({
      content: [{ type: 'text', text }],
    });

  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
// api/analyze.js
// Vercel serverless function — proxies requests to Google Gemini (free tier)
// FREE TIER: https://aistudio.google.com/apikey  (no credit card required)
// Set GEMINI_API_KEY in your Vercel project → Settings → Environment Variables

const GEMINI_MODEL = 'gemini-2.0-flash';  

export default async function handler(req, res) {
  // ── CORS ─────────────────────────────────────────────────────────────────
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST')   return res.status(405).json({ error: 'Method Not Allowed' });

  // ── Validate key ─────────────────────────────────────────────────────────
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({
      error: 'GEMINI_API_KEY is not set. Add it in Vercel → Settings → Environment Variables. Get a free key at https://aistudio.google.com/apikey'
    });
  }

  try {
    // ── Convert Anthropic-style request → Gemini format ───────────────────
    // Frontend sends: { model, max_tokens, messages: [{ role, content: [...parts] }] }
    const { messages, max_tokens } = req.body;
    const userMessage = messages?.find(m => m.role === 'user');
    if (!userMessage) return res.status(400).json({ error: 'No user message found in request body' });

    const parts = Array.isArray(userMessage.content) ? userMessage.content : [];

    // Build Gemini parts array
    const geminiParts = parts.map(part => {
      if (part.type === 'text') {
        return { text: part.text };
      }
      if (part.type === 'image') {
        // Anthropic: { source: { type:'base64', media_type, data } }
        return {
          inline_data: {
            mime_type: part.source?.media_type || 'image/jpeg',
            data:       part.source?.data       || '',
          }
        };
      }
      return null;
    }).filter(Boolean);

    // ── Call Gemini ──────────────────────────────────────────────────────
    const geminiRes = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${apiKey}`,
      {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: geminiParts }],
          generationConfig: {
            maxOutputTokens: max_tokens || 1000,
            temperature:     0.2,
          }
        })
      }
    );

    const geminiData = await geminiRes.json();

    if (!geminiRes.ok) {
      return res.status(geminiRes.status).json({
        error:   geminiData?.error?.message || 'Gemini API error',
        details: geminiData,
      });
    }

    // ── Normalize Gemini response → Anthropic shape ───────────────────────
    // So the frontend code (apiData.content[].text) keeps working unchanged.
    const text = geminiData.candidates?.[0]?.content?.parts?.[0]?.text || '';
    return res.status(200).json({
      content: [{ type: 'text', text }]
    });

  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
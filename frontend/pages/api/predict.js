export default async function handler(req, res) {
  const apiUrl = process.env.API_URL;
  const apiKey = process.env.API_KEY;

  if (!apiKey) {
    return res.status(500).json({ error: 'API Key not configured' });
  }

  // Add HTTPS check
  if (!apiUrl.startsWith('https://')) {
    return res.status(500).json({ error: 'API must be served over HTTPS' });
  }

  try {
    const response = await fetch(`${apiUrl}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': apiKey,
      },
      body: JSON.stringify(req.body),
    });

    if (!response.ok) {
      const error = await response.text();
      return res.status(response.status).json({ error: error || `HTTP error! status: ${response.status}` });
    }

    const data = await response.json();
    res.status(200).json(data);
  } catch (error) {
    console.error('Fetch error:', error);
    res.status(500).json({ error: error.message || 'Network request failed' });
  }
}

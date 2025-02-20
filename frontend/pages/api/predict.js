export default async function handler(req, res) {
  const apiUrl = process.env.API_URL;
  const apiKey = process.env.API_KEY;
  const isDevelopment = process.env.NODE_ENV === 'development';

  if (!apiKey) {
    return res.status(500).json({ error: 'API Key not configured' });
  }

  // Skip HTTPS check for local development
  if (!isDevelopment && !apiUrl.startsWith('https://')) {
    return res.status(500).json({ error: 'API must be served over HTTPS in production' });
  }

  try {
    const response = await fetch(`${apiUrl}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': apiKey,  // Always include API key
      },
      body: JSON.stringify(req.body),
    });

    const contentType = response.headers.get("content-type");
    if (!response.ok) {
      // Handle HTML error responses
      if (contentType && contentType.includes("text/html")) {
        return res.status(503).json({
          error: "Service is temporarily unavailable. Please try again later."
        });
      }

      // Handle JSON error responses
      try {
        const _ = await response.json();
        return res.status(response.status).json(_);
      } catch {
        // If can't parse JSON, return text
        await response.text(); // Remove unused variable assignment
        return res.status(response.status).json({
          error: "Service error. Please try again later."
        });
      }
    }

    const data = await response.json();
    res.status(200).json(data);
  } catch (error) {
    console.error('Fetch error:', error);
    res.status(500).json({
      error: "Unable to reach prediction service. Please try again later."
    });
  }
}

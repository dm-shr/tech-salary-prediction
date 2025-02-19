import handler from '../../api/predict';

describe('API Handler', () => {
  const mockReq = {
    body: {
      title: 'Data Scientist',
      company: 'Test Company',
      location: 'Stockholm',
      description: 'Looking for a Data Scientist',
      skills: 'python,sql',
      experience_from: 2,
      experience_to: 5,
    }
  };

  const mockRes = {
    status: jest.fn().mockReturnThis(),
    json: jest.fn()
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('rejects non-HTTPS API URLs', async () => {
    process.env.API_URL = 'http://insecure-url.com';
    process.env.API_KEY = 'test-key';

    await handler(mockReq, mockRes);

    expect(mockRes.status).toHaveBeenCalledWith(500);
    expect(mockRes.json).toHaveBeenCalledWith({
      error: 'API must be served over HTTPS'
    });
  });

  test('handles missing API key', async () => {
    process.env.API_URL = 'https://api.example.com';
    delete process.env.API_KEY;

    await handler(mockReq, mockRes);

    expect(mockRes.status).toHaveBeenCalledWith(500);
    expect(mockRes.json).toHaveBeenCalledWith({
      error: 'API Key not configured'
    });
  });

  test('successfully makes prediction request', async () => {
    const mockFetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ predicted_salary: 50000 })
    });
    global.fetch = mockFetch;

    process.env.API_URL = 'https://api.example.com';
    process.env.API_KEY = 'test-key';

    await handler(mockReq, mockRes);

    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.example.com/predict',
      expect.objectContaining({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'test-key',
        }
      })
    );
    expect(mockRes.status).toHaveBeenCalledWith(200);
  });
});

const WINDOW_SIZE = 60000; // 1 minute
const MAX_REQUESTS = 10;

export class RateLimiter {
  private requests: number[];

  constructor() {
    this.requests = [];
  }

  checkLimit(): boolean {
    const now = Date.now();
    this.requests = this.requests.filter(time => now - time < WINDOW_SIZE);

    if (this.requests.length >= MAX_REQUESTS) {
      return false;
    }

    this.requests.push(now);
    return true;
  }
}

export const rateLimiter = new RateLimiter();

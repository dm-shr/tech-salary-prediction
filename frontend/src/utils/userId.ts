import { v4 as uuidv4 } from 'uuid';

export const getUserId = (): string => {
  if (typeof window === 'undefined') {
    return ''; // Return empty string during SSR
  }

  let userId = localStorage.getItem("user_id");
  if (!userId) {
    userId = uuidv4();
    localStorage.setItem("user_id", userId);
  }
  return userId;
};

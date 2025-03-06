"use client";

import SalaryPredictorForm from "../components/SalaryPredictorForm"
import { WaitlistPanel } from "@/components/waitlist-panel";
import { useEffect, useState } from 'react';
import { getUserId } from "@/utils/userId";

export default function Home() {
  const [userId, setUserId] = useState<string>('');

  useEffect(() => {
    const id = getUserId();
    setUserId(id);
  }, []);

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-1 text-center">IT Industry Salary Prediction</h1>
      <h2 className="text-lg text-gray-700 mb-1 text-center">Get an approximate starting salary for a job opening</h2>
      <SalaryPredictorForm userId={userId} />
      <div className="md:hidden">
        <WaitlistPanel className="fixed bottom-5 right-5 z-40" />
      </div>
    </div>
  )
}

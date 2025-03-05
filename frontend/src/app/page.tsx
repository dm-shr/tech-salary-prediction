import SalaryPredictorForm from "../components/SalaryPredictorForm"
import { WaitlistPanel } from "@/components/waitlist-panel";

// Define metadata for App Router
export const metadata = {
  title: 'IT Industry Salary Prediction',
  description: 'Get an approximate starting salary for a job opening in the IT sector.',
  openGraph: {
    title: 'IT Industry Salary Prediction',
    description: 'Get an approximate starting salary for a job opening',
    images: [
      {
        url: '/opengraph-image.png',
        width: 1200,
        height: 630,
        alt: 'IT Industry Salary Prediction',
      },
    ],
    type: 'website',
    url: 'https://tech-salary-prediction.vercel.app/',
  },
}

export default function Home() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-1 text-center">IT Industry Salary Prediction</h1>
      <h2 className="text-lg text-gray-700 mb-1 text-center">Get an approximate starting salary for a job opening</h2>
      <SalaryPredictorForm />

      {/* This instance only appears on mobile */}
      <div className="md:hidden">
        <WaitlistPanel className="fixed bottom-5 right-5 z-40" />
      </div>
    </div>
  )
}

import SalaryPredictorForm from "../components/SalaryPredictorForm"
import Head from 'next/head';

export default function Home() {
  return (
    <>
      <Head>
        <title>Tech Salary Prediction</title>
        <meta name="description" content="Get an approximate starting salary for a job opening in the tech sector." />
        <meta property="og:title" content="Tech Sector Salary Prediction" />
        <meta property="og:description" content="Get an approximate starting salary for a job opening" />
        <meta property="og:image" content="/opengraph-image.png" />
        <meta property="og:url" content="https://tech-salary-prediction.vercel.app/" />
        <meta property="og:type" content="website" />
      </Head>
      <div className="container mx-auto p-4">
        <h1 className="text-2xl font-bold mb-1 text-center">Tech Sector Salary Prediction</h1>
        <h2 className="text-lg text-gray-700 mb-1 text-center">Get an approximate starting salary for a job opening</h2>
        <SalaryPredictorForm />
      </div>
    </>
  )
}

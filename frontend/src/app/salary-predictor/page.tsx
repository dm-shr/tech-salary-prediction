import SalaryPredictorForm from "./SalaryPredictorForm"

export default function SalaryPredictorPage() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-1 text-center">Tech Sector Salary Prediction</h1>
      <h2 className="text-lg text-gray-700 mb-1 text-center">Get an approximate starting salary for a job opening</h2>
      <SalaryPredictorForm />
    </div>
  )
}

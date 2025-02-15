"use client"

import type React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import { FaGithub, FaLinkedin } from "react-icons/fa";
import { fetchFromAPI } from "@/lib/api"
import { rateLimiter } from "@/lib/rateLimit"

export default function SalaryPredictorForm() {
  const [title, setTitle] = useState("Machine Learning Engineer")
  const [company, setCompany] = useState("Spotify")
  const [location, setLocation] = useState("Stockholm")
const [description, setDescription] = useState("We are seeking a Machine Learning Engineer to join our team. The ideal candidate will have experience in developing and deploying ML models, working with large datasets, and implementing end-to-end ML pipelines. Key responsibilities include model development, experimentation, and collaboration with cross-functional teams. Strong programming skills in Python and experience with deep learning frameworks required.")
  const [skills, setSkills] = useState("Python, SQL, PyTorch")
  const [experienceRange, setExperienceRange] = useState([0, 3]);
  const [predictedSalary, setPredictedSalary] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const titleExamples = ["DevOps", "Data Analyst"]
  const companyExamples = ["H&M", "Klarna", "King"]
  const locationExamples = ["Malmö", "Gothenburg"]
//   const descriptionExamples = ["We are looking for a Data Scientist!", "Join our team as a Machine Learning Engineer", "Analyze data and provide insights"]
  const skillsExamples = ["React", "AWS", "Docker"]

  const handleSkillButtonClick = (skill: string) => {
    if (skills.length === 0) {
      setSkills(skill);
    } else if (skills.includes(skill)) {
      setSkills(skills.split(',').filter(s => s !== skill).join(','));
    }
    else {
      setSkills(skills + ", " + skill);
    }
  };

  const validateInput = () => {
    if (title.length < 2) return "Job title is too short";
    if (company.length < 2) return "Company name is too short";
    if (location.length < 2) return "Location is too short";
    if (description.length < 10) return "Description is too short";
    if (skills.length < 3) return "Please add some skills";
    return null;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!rateLimiter.checkLimit()) {
      setError("Too many requests. Please wait a minute.");
      return;
    }

    setError(null)
    setPredictedSalary(null)

    const validationError = validateInput();
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsLoading(true)

    try {
      const data = await fetchFromAPI("/predict", {
        method: "POST",
        body: JSON.stringify({
          title,
          company,
          location,
          description,
          skills,
          experience_from: experienceRange[0],
          experience_to: experienceRange[1],
        }),
      })

      setPredictedSalary(data.predicted_salary)
    } catch (error) {
      console.error("Error predicting salary:", error)
      setError(error instanceof Error ? error.message : "Failed to predict salary")
      setPredictedSalary(null)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mt-2 pt-2 border-t border-gray-200 text-center mb-4"> {/* Added mb-4 */}
        <p className="text-sm text-gray-600">Interested? Let's stay in touch!</p>
        <div className="mt-2 flex justify-center space-x-4">
          <a
            href="https://github.com/dm-shr"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-900 hover:text-black flex items-center gap-2" // Changed from blue-500/600 to gray-900/black
          >
            <FaGithub className="text-xl text-black" /> {/* Added text-black */}
            <span>GitHub</span>
          </a>
          <a
            href="https://www.linkedin.com/in/dshiriaev/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-800 flex items-center gap-2" // Changed from blue-500/600 to blue-600/800
          >
            <FaLinkedin className="text-xl" />
            <span>LinkedIn</span>
          </a>
        </div>
      </div>

      <div className="border border-gray-300 rounded p-4 bg-gray-50"> {/* Changed from just 'border' to 'border border-gray-300' */}
        <form onSubmit={handleSubmit} className="space-y-8">
          <div className="mb-4">
            <Label htmlFor="title" className="mb-3 block">Job Title</Label>
            <div className="flex">
              <Input id="title" value={title} onChange={(e) => setTitle(e.target.value)} required className="w-3/5" />
              <div className="flex flex-wrap justify-center space-x-2 mt-1 w-2/5">
                {titleExamples.map((example) => (
                  <Button type="button" variant="outline" size="sm" key={example} onClick={() => setTitle(example)}
                  className="btn-outline">
                    {example}
                  </Button>
                ))}
              </div>
            </div>
          </div>
          <div className="mb-4">
            <Label htmlFor="company" className="mb-2 block">Company</Label>
            <div className="flex">
              <Input id="company" value={company} onChange={(e) => setCompany(e.target.value)} required className="w-3/5" />
              <div className="flex flex-wrap justify-center space-x-2 mt-1 w-2/5">
                {companyExamples.map((example) => (
                  <Button type="button" variant="outline" size="sm" key={example} onClick={() => setCompany(example)}
                  className="btn-outline">
                    {example}
                  </Button>
                ))}
              </div>
            </div>
          </div>
          <div className="mb-2">
            <Label htmlFor="location" className="mb-3 block">Location</Label>
            <div className="flex">
              <Input id="location" value={location} onChange={(e) => setLocation(e.target.value)} required className="w-3/5" />
              <div className="flex flex-wrap justify-center space-x-2 mt-1 w-2/5">
                {locationExamples.map((example) => (
                  <Button type="button" variant="outline" size="sm" key={example} onClick={() => setLocation(example)}
                  className="btn-outline">
                    {example}
                  </Button>
                ))}
              </div>
            </div>
          </div>
          <div className="mb-4">
            <Label htmlFor="skills" className="mb-3 block">Skills, comma-separated</Label>
            <div className="flex">
              <Input id="skills" value={skills} onChange={(e) => setSkills(e.target.value)} required className="w-3/5" />
              <div className="flex flex-wrap justify-center space-x-2 mt-1 w-2/5">
                {skillsExamples.map((skill) => (
                  <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  key={skill}
                  onClick={() => handleSkillButtonClick(skill)}
                  className="btn-outline"
                  >
                    {skill}
                  </Button>
                ))}
              </div>
            </div>
          </div>
          <div className="mb-4 flex flex-col">
            <Label htmlFor="experienceRange" className="mb-6 block">Experience Range, years</Label>
            <div className="w-4/5 mx-auto"> {/* Added container with width constraint */}
              <Slider
                min={0}
                max={10}
                step={1}
                range
                value={experienceRange}
                onChange={(value) => setExperienceRange(value as [number, number])}
              />
              <div className="flex justify-between mt-2">
                <div>{experienceRange[0]} years</div>
                <div>{experienceRange[1]} years</div>
              </div>
            </div>
          </div>
          <div className="mb-4">
            <Label htmlFor="description" className="mb-4 block">Job Description</Label>
            <div className="flex">
              <Textarea id="description" value={description} onChange={(e) => setDescription(e.target.value)} required className="w-full" />
              {/* <div className="flex flex-wrap justify-center space-x-2 mt-1 w-2/5"> */}
              <div className="flex flex-wrap justify-center space-x-2 mt-1 w-2/7">
                {/* {descriptionExamples.map((example) => (
                <Button type="button" variant="outline" size="sm" key={example} onClick={() => setDescription(example)}>
                  {example}
                </Button>
              ))} */}
              </div>
            </div>
          </div>
          <div className="flex flex-col items-center justify-center mt-6">
            <div className="w-full max-w-xs">
              <Button
                type="submit"
                disabled={isLoading}
                className="w-full h-10 flex items-center justify-center btn-primary"
              >
                {isLoading ? "Predicting..." : "Predict Salary"}
              </Button>
            </div>
            <div className="h-12 w-full flex items-center justify-center mt-6"> {/* Changed mt-8 to mt-6 */}
              {error ? (
                <div className="px-4 py-3 rounded-lg shadow-md w-full max-w-sm bg-red-50 border border-red-200">
                  <p className="text-red-700 text-center">{error}</p>
                </div>
              ) : predictedSalary && (
                <div className="px-4 py-3 rounded-lg shadow-md w-full max-w-sm bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-100"> {/* Changed p-4 to px-4 py-3 */}
                  <p className="text-gray-800 text-lg text-center flex items-center justify-center gap-2"> {/* Removed font-semibold */}
                    Starting Salary:
                    <span className="text-indigo-700 font-medium">{predictedSalary}</span> {/* Changed from default bold to font-medium */}
                  </p>
                </div>
              )}
            </div>
          </div>
        </form>
      </div>
    </div>
  )
}

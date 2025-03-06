"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import { FaGithub, FaLinkedin } from "react-icons/fa";
import { MdOutlineContactPage } from "react-icons/md";
import { rateLimiter } from "@/utils/rateLimit"
import { WaitlistPanel } from "./waitlist-panel"

const CURRENCY_CONVERSION = 0.38; // Define the currency conversion rate

interface SalaryPredictorFormProps {
  userId: string;
}

export default function SalaryPredictorForm({ userId }: SalaryPredictorFormProps) {
  const [title, setTitle] = useState("Machine Learning Engineer")
  const [company, setCompany] = useState("Spotify")
  const [location, setLocation] = useState("Stockholm")
  const [description, setDescription] = useState("We are seeking a Machine Learning Engineer to join our team. The ideal candidate will have experience in developing and deploying ML models, working with large datasets, and implementing end-to-end ML pipelines. Key responsibilities include model development, experimentation, and collaboration with cross-functional teams. Strong programming skills in Python and experience with deep learning frameworks required.")
  const [skills, setSkills] = useState("Python, SQL, PyTorch")
  const [experienceRange, setExperienceRange] = useState([3, 6]);
  const [predictedSalary, setPredictedSalary] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const titleExamples = ["DevOps", "Data Analyst"]
  const companyExamples = ["H&M", "Klarna", "King"]
  const locationExamples = ["Malmö", "Gothenburg"]
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

    if (!userId) {
      console.error("User ID not available");
      setError("User identification error. Please try reloading the page.");
      return;
    }

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
      const response = await fetch("/api/predict", { // Call the Vercel Function
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          userId, // Include the userId in the request
          title,
          company,
          location,
          description,
          skills,
          experience_from: experienceRange[0],
          experience_to: experienceRange[1],
        }),
      })

      const data = await response.json();

      if (!response.ok) {
        setError(data.error || "An unexpected error occurred");
        setPredictedSalary(null);
        return;
      }

      const predictedSalaryValue = parseFloat(data.predicted_salary) * CURRENCY_CONVERSION;
      const roundedSalary = Math.round(predictedSalaryValue / 100) * 100;
      const formattedSalary = roundedSalary.toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
      setPredictedSalary(`${formattedSalary} SEK/month`);
    } catch (error) {
      console.error("Error predicting salary:", error)
      setError("Unable to get prediction. Please try again later.")
      setPredictedSalary(null)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (experienceRange[1] < experienceRange[0]) {
      setExperienceRange([experienceRange[0], experienceRange[0]]);
    }
  }, [experienceRange]);

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mt-2 pt-2 border-t border-gray-200 text-center mb-4">
        <p className="text-sm text-gray-600">Interested? Let&apos;s stay in touch!</p>
        <div className="mt-2 flex justify-center space-x-4">
          <a
            href="https://github.com/dm-shr"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-900 hover:text-black flex items-center gap-2"
          >
            <FaGithub className="text-xl text-black" />
            <span className="hidden md:inline">GitHub</span>
          </a>
          <a
            href="https://www.linkedin.com/in/dshiriaev/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-800 flex items-center gap-2"
          >
            <FaLinkedin className="text-xl" />
            <span className="hidden md:inline">LinkedIn</span>
          </a>
          <a
            href="https://shiriaev.vercel.app/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-900 hover:text-black flex items-center gap-2"
          >
            <MdOutlineContactPage className="text-xl" />
            <span className="hidden md:inline">Portfolio</span>
          </a>
        </div>
      </div>

      <div className="border border-gray-300 rounded p-4 bg-gray-50">
        <form onSubmit={handleSubmit} className="space-y-8">
          <div className="mb-4">
            <Label htmlFor="title" className="mb-3 block">Job Title</Label>
            <div className="flex flex-col md:flex-row"> {/* Switch to row layout on medium screens and up */}
              <Input id="title" value={title} onChange={(e) => setTitle(e.target.value)} required className="w-full md:w-3/5" /> {/* Adjust width on medium screens and up */}
              <div className="flex flex-wrap justify-center space-x-2 mt-1 w-full md:w-2/5"> {/* Adjust width on medium screens and up */}
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
            <div className="flex flex-col md:flex-row"> {/* Switch to row layout on medium screens and up */}
              <Input id="company" value={company} onChange={(e) => setCompany(e.target.value)} required className="w-full md:w-3/5" /> {/* Adjust width on medium screens and up */}
              <div className="flex flex-wrap justify-center space-x-2 mt-1 w-full md:w-2/5"> {/* Adjust width on medium screens and up */}
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
            <div className="flex flex-col md:flex-row"> {/* Switch to row layout on medium screens and up */}
              <Input id="location" value={location} onChange={(e) => setLocation(e.target.value)} required className="w-full md:w-3/5" /> {/* Adjust width on medium screens and up */}
              <div className="flex flex-wrap justify-center space-x-2 mt-1 w-full md:w-2/5"> {/* Adjust width on medium screens and up */}
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
            <div className="flex flex-col md:flex-row"> {/* Switch to row layout on medium screens and up */}
              <Input id="skills" value={skills} onChange={(e) => setSkills(e.target.value)} required className="w-full md:w-3/5" /> {/* Adjust width on medium screens and up */}
              <div className="flex flex-wrap justify-center space-x-2 mt-1 w-full md:w-2/5"> {/* Adjust width on medium screens and up */}
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
              <style jsx global>{`
                .rc-slider-rail {
                  height: 12px !important; /* Adjust the height of the rail */
                }
                .rc-slider-track {
                  height: 12px !important; /* Adjust the height of the track */
                  background-color: #b2aee7 !important; /* Change track color to #8781dc */
                }
                .rc-slider-handle {
                  width: 24px !important; /* Adjust the width of the handle */
                  height: 24px !important; /* Adjust the height of the handle */
                  margin-top: -6px !important; /* Adjust the margin to center the handle */
                  border: solid 2px #b2aee7 !important; /* Change handle border color to match */
                }
                .rc-slider-handle:active {
                  border-color: #6259d2 !important; /* Slightly darker color when active */
                  box-shadow: 0 0 5px #b2aee7 !important; /* Add glow effect when active */
                }
                .rc-slider-handle:hover {
                  border-color: #6259d2 !important; /* Slightly darker on hover */
                }
              `}</style>
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

          <div className="hidden md:flex relative -right-10 justify-end mb-4">
            <WaitlistPanel />
          </div>

          <div className="mb-4">
            <Label htmlFor="description" className="mb-4 block">Job Description</Label>
            <div className="flex flex-col">
              <Textarea id="description" value={description} onChange={(e) => setDescription(e.target.value)} required className="w-full" />
              <div className="flex flex-wrap justify-center space-x-2 mt-1 w-full">
              </div>
            </div>
          </div>

          <div className="flex flex-col items-center justify-center mt-6">
            <div className="w-full max-w-xs" style={{ position: 'relative', top: '-20px' }}>
              <Button
                type="submit"
                disabled={isLoading}
                className="w-full h-10 flex items-center justify-center btn-primary"
              >
                {isLoading ? "Predicting..." : "Predict Salary"}
              </Button>
            </div>
            <div className="h-12 w-full flex items-center justify-center mt-6" style={{ position: 'relative', top: '-25px' }}>
              {error ? (
                <div className="px-4 py-3 rounded-lg shadow-md w-full max-w-sm bg-red-50 border border-red-200">
                  <p className="text-red-700 text-center">{error}</p>
                </div>
              ) : predictedSalary && (
                <div className="px-4 py-3 rounded-lg shadow-md w-full max-w-sm bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-100">
                  <p className="text-gray-800 text-lg text-center flex items-center justify-center gap-2">
                    Starting Salary:
                    <span className="text-indigo-700 font-medium">{predictedSalary}</span>
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

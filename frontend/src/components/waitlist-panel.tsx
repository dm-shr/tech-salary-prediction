"use client";

import { useState, useEffect } from "react";
import { X, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import Script from "next/script";

type WaitlistPanelProps = {
  className?: string; // Add className prop for custom positioning
};

export function WaitlistPanel({ className = "" }: WaitlistPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isLocalEnvironment, setIsLocalEnvironment] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    // Check if we're in a local development environment
    const hostname = window.location.hostname;
    setIsLocalEnvironment(hostname === 'localhost' || hostname === '127.0.0.1');

    // Check if device is mobile (screen width < 768px)
    const checkIfMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };

    // Initial check
    checkIfMobile();

    // Add resize listener
    window.addEventListener('resize', checkIfMobile);

    // Clean up listener
    return () => {
      window.removeEventListener('resize', checkIfMobile);
    };
  }, []);

  useEffect(() => {
    // Add a global error handler for the waitlist widget
    const originalErrorHandler = window.onerror;
    window.onerror = function(message, source, lineno, colno, error) {
      if (message.toString().includes('postWaiterObj') ||
          (source && source.includes('getwaitlist.min.js'))) {
        setErrorMessage("Waitlist form submission isn't fully functional in local development. Please try on the production site.");
        return true; // Prevent default error handling
      }
      // Call the original error handler for other errors
      if (originalErrorHandler) {
        return originalErrorHandler(message, source, lineno, colno, error);
      }
      return false;
    };

    // Add a mutation observer to modify widget buttons once loaded
    if (isOpen) {
      const observer = new MutationObserver(() => {
        const widgetContainer = document.getElementById('getWaitlistContainer');
        if (widgetContainer) {
          // Find all buttons inside the widget
          const widgetButtons = widgetContainer.querySelectorAll('button');

          // Set type="button" on all buttons to prevent form submission
          widgetButtons.forEach(button => {
            button.setAttribute('type', 'button');
          });

          // Also prevent form submission when clicking inside the widget
          widgetContainer.addEventListener('click', (e) => {
            e.stopPropagation();
          });
        }
      });

      // Start observing the document with the configured parameters
      observer.observe(document.body, { childList: true, subtree: true });

      return () => {
        // Cleanup the observer when component unmounts or panel closes
        observer.disconnect();
      };
    }

    return () => {
      // Restore original error handler when component unmounts
      window.onerror = originalErrorHandler;
    };
  }, [isOpen]); // Added isOpen to the dependency array

  // Inject CSS for waitlist widget
  const cssLoader = `
    let head = document.getElementsByTagName('HEAD')[0];
    let link = document.createElement('link');
    link.rel = 'stylesheet';
    link.type = 'text/css';
    link.href = 'https://prod-waitlist-widget.s3.us-east-2.amazonaws.com/getwaitlist.min.css';
    head.appendChild(link);
  `;

  // Add script to prevent form submission from widget
  const preventFormSubmission = `
    // This ensures buttons inside the waitlist widget don't trigger form submission
    window.addEventListener('load', function() {
      const observer = new MutationObserver(function(mutations) {
        const widgetContainer = document.getElementById('getWaitlistContainer');
        if (widgetContainer) {
          const forms = widgetContainer.querySelectorAll('form');
          forms.forEach(form => {
            form.onsubmit = function(e) {
              e.stopPropagation();
            }
          });
        }
      });

      observer.observe(document.body, { childList: true, subtree: true });
    });
  `;

  return (
    <>
      {/* Button wrapper with conditional positioning */}
      <div className={`${isMobile ? 'fixed bottom-5 right-5 z-40' : ''} ${className}`}>
        <Button
          onClick={() => setIsOpen(true)}
          type="button"
          className="bg-[#6259d2] hover:bg-[#6517b4] text-white px-4 py-2 text-sm rounded-lg shadow-md"
          size="sm"
        >
          Get Early Access
        </Button>
      </div>

      {/* Sliding panel */}
      <div
        className={`fixed top-0 right-0 h-full w-80 bg-white shadow-lg transform transition-transform duration-300 ease-in-out z-50 ${
          isOpen ? "translate-x-0" : "translate-x-full"
        }`}
        onClick={(e) => e.stopPropagation()} // Stop click propagation
      >
        <div className="p-4 h-full flex flex-col">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-bold text-lg">Join Our Waitlist</h3>
            <Button
              onClick={() => setIsOpen(false)}
              type="button" // Add type="button" to prevent form submission
              variant="ghost"
              size="icon"
              className="rounded-full hover:bg-[#f0e6fc] hover:text-[#761ad6]"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>

          {isLocalEnvironment && (
            <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md text-yellow-800 text-sm">
              <div className="flex items-start">
                <AlertCircle className="h-4 w-4 mr-2 mt-0.5 flex-shrink-0" />
                <p>
                  You&apos;re in a local development environment. The waitlist form might not work correctly here. For best results, use the production site.
                </p>
              </div>
            </div>
          )}

          {errorMessage && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md text-red-800 text-sm">
              <div className="flex items-start">
                <AlertCircle className="h-4 w-4 mr-2 mt-0.5 flex-shrink-0" />
                <p>{errorMessage}</p>
              </div>
            </div>
          )}

          <div className="flex-grow overflow-y-auto">
            {/* Waitlist widget */}
            {isOpen && (
              <>
                <Script id="waitlist-css-loader" dangerouslySetInnerHTML={{ __html: cssLoader }} />
                <Script id="prevent-form-submission" dangerouslySetInnerHTML={{ __html: preventFormSubmission }} />
                <Script src="https://prod-waitlist-widget.s3.us-east-2.amazonaws.com/getwaitlist.min.js" />
                <div
                  id="getWaitlistContainer"
                  data-waitlist_id="25902"
                  data-widget_type="WIDGET_1"
                  onClick={(e) => e.stopPropagation()} // Stop click propagation
                ></div>
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

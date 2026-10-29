// Number dialled by the emergency banner. 112 reaches emergency services from most
// mobile phones worldwide; change it for your region if needed.
export const EMERGENCY_NUMBER = "112";

// Quick replies fill the composer; the bracketed part is selected for the user to replace.
export const QUICK_REPLIES = [
  { id: "symptoms", label: "Check symptoms", template: "What are the common symptoms of [condition]?" },
  { id: "medication", label: "Medication info", template: "What is [medication] used for, and what are its side effects?" },
  { id: "visit", label: "Prepare for doctor visit", template: "How is [condition] diagnosed?" },
  { id: "causes", label: "Understand causes", template: "What causes [condition]?" },
  { id: "treatment", label: "Treatment options", template: "What are the treatment options for [condition]?" },
];

export const MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024;

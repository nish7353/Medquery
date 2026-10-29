export function LogoMark({ className = "h-9 w-9" }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
      <circle cx="16" cy="16" r="16" fill="#1F2A1C" />
      <path d="M14 9h4v5h5v4h-5v5h-4v-5H9v-4h5z" fill="#C2CB52" />
    </svg>
  );
}

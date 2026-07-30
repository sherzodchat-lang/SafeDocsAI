import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { LocaleProvider } from './i18n'
import { SourcesProvider } from './contexts/SourcesContext'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <LocaleProvider>
      <SourcesProvider>
        <App />
      </SourcesProvider>
    </LocaleProvider>
  </StrictMode>,
)

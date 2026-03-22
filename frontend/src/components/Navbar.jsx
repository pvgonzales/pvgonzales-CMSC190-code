import { useNavigate, useLocation } from 'react-router-dom';

export default function Navbar() {
  const navigate = useNavigate();
  const location = useLocation();
  const isLanding = location.pathname === '/';

  const handleNavClick = (hash) => {
    if (isLanding) {
      const el = document.getElementById(hash);
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    } else {
      navigate('/#' + hash);
    }
  };

  return (
    <nav className="navbar">
      <div className="navbar-brand" onClick={() => navigate('/')}>
        CMSC 190 - 2
      </div>

      <div className="navbar-links">
        <button className="nav-link" onClick={() => handleNavClick('home')}>
          Home
        </button>
        <button className="nav-link" onClick={() => handleNavClick('about')}>
          About the Study
        </button>
        <button className="nav-link" onClick={() => handleNavClick('methods')}>
          Methods
        </button>
      </div>

      <button className="navbar-cta" onClick={() => navigate('/analyze')}>
        Analyze Footage →
      </button>
    </nav>
  );
}

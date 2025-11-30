import React from 'react';

export const Header: React.FC = () => {
  return (
    <header className="header">
      <div className="header-brand">
        <img
          src="/src/logo/logo.png"
          alt="Text2VR Logo"
          className="header-logo"
        />
        <div>
          <h1 className="header-title">Text2VR</h1>
          <p className="header-subtitle">Generate VR environments from text</p>
        </div>
      </div>
    </header>
  );
};

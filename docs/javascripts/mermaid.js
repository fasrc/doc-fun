document$.subscribe(({ body }) => {
  mermaid.initialize({ 
    startOnLoad: true,
    theme: 'default',
    themeVariables: {
      primaryColor: '#2196f3',
      primaryTextColor: '#fff',
      primaryBorderColor: '#1976d2',
      lineColor: '#757575',
      secondaryColor: '#f5f5f5',
      tertiaryColor: '#e3f2fd'
    }
  });
  mermaid.run({
    querySelector: '.mermaid'
  });
});
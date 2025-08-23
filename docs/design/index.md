# Design Documentation

This section contains architectural decisions, design specifications, and technical documentation for the doc-generator project.

## 📐 Design Principles

The doc-generator project follows these core design principles:

### **Modularity**
- Plugin-based architecture for extensibility
- Provider abstraction for multiple AI services (OpenAI, Claude)
- Separation of concerns between generation, extraction, standardization, analysis, and evaluation
- Content extraction system with extensible format support

### **Backward Compatibility**
- New features don't break existing functionality
- Configuration changes are additive
- API changes follow semantic versioning

### **Quality Focus**
- Multi-run generation with quality evaluation
- Comprehensive testing at unit and integration levels
- Documentation-driven development

### **User Experience**
- Simple CLI for basic usage
- Advanced options for power users
- Clear error messages and helpful defaults

---

## 📋 Design Documents

| Document | Status | Description |
|----------|--------|-------------|
| [Claude API Integration](claude-api-integration.md) | ✅ Implemented | Adding Anthropic Claude alongside OpenAI |
| Document Standardization System | ✅ Implemented | Architecture for transforming existing documentation |
| Content Extraction Framework | ✅ Implemented | Extensible system for parsing various document formats |
| README Generation Pipeline | ✅ Implemented | Automated README creation for code projects |
| Multi-Provider Strategy | ✅ Implemented | Framework for supporting multiple LLM providers |

---

## 🎯 Architecture Overview

```mermaid
graph TB
    A[CLI Interface] --> B[Core Generation]
    A --> C[README Generation] 
    A --> D[Document Standardization]
    A --> E[Legacy Code Scanning]
    
    B --> F[DocumentationGenerator]
    C --> G[ReadmeGenerator]
    D --> H[DocumentStandardizer]
    
    F --> I[Provider Manager]
    G --> I
    H --> I
    I --> J[OpenAI Provider]
    I --> K[Claude Provider]
    
    H --> L[Content Extractors]
    L --> M[HTML Extractor]
    L --> N[Future Extractors]
    
    H --> O[Section Mapper]
    O --> P[Template Engine]
    
    F --> Q[Plugin Manager]
    Q --> R[ModuleRecommender]
    Q --> S[Analysis Plugins]
    
    F --> T[Quality Pipeline]
    T --> U[DocumentAnalyzer]
    T --> V[GPTQualityEvaluator]
    
    style N fill:#f9f,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    style S fill:#f9f,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
```

*Dashed components represent extensible plugin points*

---

## 📝 Decision Records

### **ADR-001: Plugin Architecture**
- **Status**: Implemented
- **Decision**: Use Python entry points for plugin discovery
- **Rationale**: Standard Python mechanism, automatic discovery, easy packaging

### **ADR-002: Provider Abstraction**
- **Status**: Implemented
- **Decision**: Create provider abstraction layer for multiple LLM services
- **Rationale**: Support multiple AI providers (OpenAI, Claude) without breaking existing code

### **ADR-004: Document Standardization Architecture**
- **Status**: Implemented  
- **Decision**: Separate content extraction from transformation logic
- **Rationale**: Enables support for multiple input formats and extensible processing

### **ADR-005: Template-Based Section Mapping**
- **Status**: Implemented
- **Decision**: Use organizational templates with AI-powered content mapping
- **Rationale**: Provides consistent structure while preserving content fidelity

### **ADR-003: Configuration Management**
- **Status**: Implemented
- **Decision**: YAML-based configuration with environment variable overrides
- **Rationale**: Human-readable, version-controllable, flexible

---

## 🔄 Design Process

Our design process follows these steps:

1. **Problem Identification** - Document the issue or feature need
2. **Requirements Gathering** - Define functional and non-functional requirements
3. **Architecture Design** - Create high-level design with alternatives
4. **Technical Specification** - Detailed implementation plan
5. **Review & Feedback** - Team review and community input
6. **Implementation** - Develop with tests and documentation
7. **Post-Implementation Review** - Evaluate outcomes and lessons learned

---

## 📚 References

- [Architectural Decision Records (ADRs)](https://adr.github.io/)
- [Design by Contract](https://en.wikipedia.org/wiki/Design_by_contract)
- [Plugin Architecture Patterns](https://www.martinfowler.com/articles/plugins.html)
- [Provider Pattern](https://en.wikipedia.org/wiki/Provider_model)
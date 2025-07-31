# Plugin System API

The plugin system provides extensible recommendation engines through a well-defined interface.

## PluginManager

::: doc_generator.plugin_manager.PluginManager
    options:
      show_source: true
      show_root_heading: true
      show_root_members_full_path: false
      show_category_heading: true
      heading_level: 3

## RecommendationEngine (Base Class)

::: doc_generator.plugins.base.RecommendationEngine  
    options:
      show_source: true
      show_root_heading: true
      show_root_members_full_path: false
      show_category_heading: true
      heading_level: 3

## Built-in Plugins

### ModuleRecommender

::: doc_generator.plugins.modules.ModuleRecommender
    options:
      show_source: true
      show_root_heading: true  
      show_root_members_full_path: false
      show_category_heading: true
      heading_level: 4
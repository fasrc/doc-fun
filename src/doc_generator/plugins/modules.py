"""
Built-in HPC module recommendation plugin.
"""

from typing import List, Dict, Optional
from .base import RecommendationEngine


class ModuleRecommender(RecommendationEngine):
    """
    Recommendation engine for HPC modules based on topic analysis.
    
    This plugin analyzes topic keywords and recommends relevant HPC modules
    from the terminology configuration, with priority given to latest FASRC builds.
    """
    
    def __init__(self, terminology: Optional[Dict] = None, **kwargs):
        super().__init__(terminology, **kwargs)
        
        # Extract HPC modules from terminology
        self.hpc_modules = self.terminology.get('hpc_modules', [])
        
        # Keyword mappings for different technologies
        self.keyword_mappings = {
            'python': ['python', 'py', 'jupyter', 'anaconda', 'conda', 'numpy', 'scipy', 'pandas'],
            'gcc': ['c', 'cpp', 'c++', 'gnu', 'gcc', 'fortran', 'gfortran'],
            'cuda': ['cuda', 'gpu', 'nvidia', 'cupy', 'numba'],
            'openmpi': ['mpi', 'openmpi', 'parallel', 'distributed'],
            'intel': ['intel', 'mkl', 'icc', 'ifort'],
            'matlab': ['matlab', 'octave'],
            'r': ['r', 'rstudio', 'cran'],
            'java': ['java', 'jvm', 'scala', 'maven'],
            'julia': ['julia', 'julialang'],
            'tensorflow': ['tensorflow', 'tf', 'keras'],
            'pytorch': ['pytorch', 'torch'],
            'singularity': ['singularity', 'container', 'apptainer'],
            'cmake': ['cmake', 'build', 'compilation'],
            'git': ['git', 'version', 'control'],
            'hdf5': ['hdf5', 'hdf', 'hierarchical'],
            'netcdf': ['netcdf', 'climate', 'atmospheric']
        }
        
        # Stop words to filter out
        self.stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'through', 'during',
            'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should',
            'now', 'use', 'using', 'used', 'get', 'getting', 'got', 'make', 'making',
            'made', 'take', 'taking', 'took', 'come', 'coming', 'came', 'go', 'going',
            'went', 'see', 'seeing', 'saw', 'know', 'knowing', 'knew', 'think',
            'thinking', 'thought', 'say', 'saying', 'said', 'work', 'working', 'worked'
        }
    
    def get_name(self) -> str:
        return "modules"
    
    def get_supported_types(self) -> List[str]:
        return ["hpc_modules", "software", "compilers", "libraries"]
    
    def get_priority(self) -> int:
        return 100  # High priority for HPC modules
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        """
        Get HPC module recommendations for the given topic.
        
        Args:
            topic: The topic to analyze for module recommendations
            context: Optional context (unused currently)
            
        Returns:
            List of recommended modules with metadata
        """
        if not self.hpc_modules:
            self.logger.warning("No HPC modules available in terminology")
            return []
        
        # Extract keywords from topic
        topic_keywords = self._extract_keywords_from_topic(topic)
        self.logger.debug(f"Extracted keywords from '{topic}': {topic_keywords}")
        
        # Find matching modules
        matching_modules = []
        for module in self.hpc_modules:
            relevance_score = self._calculate_module_relevance(module, topic_keywords)
            if relevance_score > 0:
                module_copy = module.copy()
                module_copy['relevance_score'] = relevance_score
                module_copy['load_command'] = f"module load {module['name']}"
                matching_modules.append(module_copy)
        
        # Sort by relevance score and priority
        matching_modules.sort(
            key=lambda x: (x['relevance_score'], self._get_priority_score(x)),
            reverse=True
        )
        
        # Return top matches (default max 3)
        max_modules = context.get('max_modules', 3) if context else 3
        return matching_modules[:max_modules]
    
    def get_formatted_recommendations(self, topic: str, context: Optional[Dict] = None) -> str:
        """
        Get formatted module recommendations for documentation context.
        """
        modules = self.get_recommendations(topic, context)
        
        if not modules:
            return ""
        
        formatted = "\n## Recommended Modules:\n\n"
        for module in modules:
            formatted += f"**{module['name']}**\n"
            formatted += f"- Load Command: `{module['load_command']}`\n"
            formatted += f"- Description: {module.get('description', 'No description available')}\n"
            if 'category' in module:
                formatted += f"- Category: {module['category']}\n"
            formatted += f"- Relevance Score: {module['relevance_score']}\n\n"
        
        return formatted
    
    def _extract_keywords_from_topic(self, topic: str) -> List[str]:
        """Extract relevant keywords from topic string."""
        import re
        
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', topic.lower())
        
        # Filter out stop words and short words
        keywords = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return keywords
    
    def _calculate_module_relevance(self, module: Dict, keywords: List[str]) -> float:
        """Calculate relevance score for a module given topic keywords."""
        score = 0.0
        module_name = module.get('name', '').lower()
        module_desc = module.get('description', '').lower()
        
        # Check each keyword mapping
        for tech, tech_keywords in self.keyword_mappings.items():
            # If any topic keyword matches this technology
            if any(keyword in tech_keywords for keyword in keywords):
                # Check if module is related to this technology
                if tech in module_name or any(tk in module_name for tk in tech_keywords):
                    score += 10.0  # Strong match
                elif any(tk in module_desc for tk in tech_keywords):
                    score += 5.0   # Weaker match
        
        # Direct keyword matches in module name (higher weight)
        for keyword in keywords:
            if keyword in module_name:
                score += 8.0
        
        # Direct keyword matches in description (lower weight)  
        for keyword in keywords:
            if keyword in module_desc:
                score += 2.0
        
        # Special case for R modules with statistics keywords
        if any(keyword in ['statistics', 'statistical', 'stats', 'data', 'analysis'] 
               for keyword in keywords):
            if 'r/' in module_name.lower() or 'rstudio' in module_name.lower():
                score += 15.0
        
        return score
    
    def _get_priority_score(self, module: Dict) -> float:
        """Calculate priority score for module ordering."""
        score = 0.0
        module_name = module.get('name', '').lower()
        
        # Prefer FASRC builds (fasrc02 > fasrc01)
        if 'fasrc02' in module_name:
            score += 2.0
        elif 'fasrc01' in module_name:
            score += 1.0
        
        # Prefer certain categories
        category = module.get('category', '').lower()
        if category in ['programming', 'compiler']:
            score += 1.0
        elif category in ['library', 'tool']:
            score += 0.5
        
        return score
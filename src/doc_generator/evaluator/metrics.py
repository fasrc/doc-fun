"""Similarity metrics and scoring for documentation comparison."""

import re
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
import numpy as np
from collections import Counter
import math


class SimilarityMetrics:
    """Calculate various similarity metrics between documents."""
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return SimilarityMetrics.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def sequence_similarity(s1: str, s2: str) -> float:
        """Calculate sequence similarity using difflib."""
        return SequenceMatcher(None, s1, s2).ratio()
    
    @staticmethod
    def jaccard_similarity(text1: str, text2: str, use_words: bool = True) -> float:
        """Calculate Jaccard similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            use_words: If True, compare words; if False, compare characters
            
        Returns:
            Jaccard similarity coefficient (0-1)
        """
        if use_words:
            set1 = set(text1.lower().split())
            set2 = set(text2.lower().split())
        else:
            set1 = set(text1.lower())
            set2 = set(text2.lower())
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    @staticmethod
    def cosine_similarity(text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using TF-IDF."""
        # Tokenize
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        # Create vocabulary
        vocabulary = list(set(words1 + words2))
        
        # Create vectors
        vector1 = [words1.count(word) for word in vocabulary]
        vector2 = [words2.count(word) for word in vocabulary]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(b * b for b in vector2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    @staticmethod
    def structural_similarity(sections1: List[Dict], sections2: List[Dict]) -> float:
        """Compare document structure (sections, hierarchy)."""
        if not sections1 and not sections2:
            return 1.0
        if not sections1 or not sections2:
            return 0.0
        
        # Extract section titles
        titles1 = [s['title'].lower() for s in sections1]
        titles2 = [s['title'].lower() for s in sections2]
        
        # Calculate title similarity
        title_sim = SimilarityMetrics.sequence_similarity(
            ' '.join(titles1), 
            ' '.join(titles2)
        )
        
        # Compare section count
        count_sim = 1 - abs(len(sections1) - len(sections2)) / max(len(sections1), len(sections2))
        
        # Compare hierarchy levels
        levels1 = [s.get('level', 1) for s in sections1]
        levels2 = [s.get('level', 1) for s in sections2]
        
        level_sim = 1.0
        if levels1 and levels2:
            avg_level1 = sum(levels1) / len(levels1)
            avg_level2 = sum(levels2) / len(levels2)
            max_level = max(max(levels1), max(levels2))
            level_sim = 1 - abs(avg_level1 - avg_level2) / max_level if max_level > 0 else 1.0
        
        # Weighted average
        return (title_sim * 0.5 + count_sim * 0.25 + level_sim * 0.25)
    
    @staticmethod
    def code_similarity(examples1: List[Dict], examples2: List[Dict]) -> float:
        """Compare code examples between documents."""
        if not examples1 and not examples2:
            return 1.0
        if not examples1 or not examples2:
            return 0.0
        
        # Extract all code
        code1 = ' '.join([ex.get('code', '') for ex in examples1])
        code2 = ' '.join([ex.get('code', '') for ex in examples2])
        
        if not code1 and not code2:
            return 1.0
        if not code1 or not code2:
            return 0.0
        
        # Normalize code (remove extra whitespace, comments)
        code1_normalized = SimilarityMetrics._normalize_code(code1)
        code2_normalized = SimilarityMetrics._normalize_code(code2)
        
        # Calculate similarity
        return SimilarityMetrics.sequence_similarity(code1_normalized, code2_normalized)
    
    @staticmethod
    def _normalize_code(code: str) -> str:
        """Normalize code for comparison."""
        # Remove comments (simple approach)
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)  # Python comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)  # C-style comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Block comments
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        
        return code.strip()
    
    @staticmethod
    def semantic_similarity(text1: str, text2: str) -> float:
        """Calculate semantic similarity using keyword overlap and ngrams."""
        # Extract important keywords (simple approach - could use TF-IDF)
        keywords1 = SimilarityMetrics._extract_keywords(text1)
        keywords2 = SimilarityMetrics._extract_keywords(text2)
        
        # Calculate keyword overlap
        keyword_sim = SimilarityMetrics.jaccard_similarity(
            ' '.join(keywords1), 
            ' '.join(keywords2)
        )
        
        # Calculate n-gram similarity
        ngrams1 = SimilarityMetrics._get_ngrams(text1, 3)
        ngrams2 = SimilarityMetrics._get_ngrams(text2, 3)
        
        ngram_sim = len(ngrams1.intersection(ngrams2)) / max(len(ngrams1), len(ngrams2)) if ngrams1 or ngrams2 else 0
        
        # Weighted average
        return (keyword_sim * 0.6 + ngram_sim * 0.4)
    
    @staticmethod
    def _extract_keywords(text: str, top_n: int = 20) -> List[str]:
        """Extract top keywords from text."""
        # Simple keyword extraction based on frequency
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were',
                     'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                     'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
        
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Get frequency
        word_freq = Counter(words)
        
        # Return top N words
        return [word for word, _ in word_freq.most_common(top_n)]
    
    @staticmethod
    def _get_ngrams(text: str, n: int) -> set:
        """Get n-grams from text."""
        words = text.lower().split()
        ngrams = set()
        
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.add(ngram)
        
        return ngrams
    
    @staticmethod
    def calculate_composite_score(
        content_sim: float,
        structural_sim: float,
        code_sim: float,
        semantic_sim: float,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate weighted composite similarity score.
        
        Args:
            content_sim: Content similarity score
            structural_sim: Structural similarity score
            code_sim: Code similarity score
            semantic_sim: Semantic similarity score
            weights: Optional custom weights
            
        Returns:
            Composite score (0-1)
        """
        if weights is None:
            weights = {
                'content': 0.3,
                'structure': 0.2,
                'code': 0.25,
                'semantic': 0.25
            }
        
        total_weight = sum(weights.values())
        
        score = (
            content_sim * weights.get('content', 0.3) +
            structural_sim * weights.get('structure', 0.2) +
            code_sim * weights.get('code', 0.25) +
            semantic_sim * weights.get('semantic', 0.25)
        ) / total_weight
        
        return min(1.0, max(0.0, score))
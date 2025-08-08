#!/usr/bin/env python3
"""
Demo script for documentation comparison module.

This script demonstrates how to:
1. Download existing documentation from a URL
2. Generate new documentation on the same topic
3. Compare the two and evaluate quality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from doc_generator.evaluator import (
    DocumentationDownloader,
    DocumentationComparator,
    SimilarityMetrics
)
from doc_generator.core import DocumentationGenerator


def demo_basic_comparison():
    """Demonstrate basic documentation comparison."""
    
    print("="*60)
    print("DOCUMENTATION COMPARISON DEMO")
    print("="*60)
    
    # Example: Compare with Python documentation
    topic = "Python List Comprehensions"
    reference_url = "https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions"
    
    print(f"\nTopic: {topic}")
    print(f"Reference URL: {reference_url}\n")
    
    # Initialize comparator
    comparator = DocumentationComparator()
    
    try:
        # Run comparison
        print("Generating documentation and comparing...")
        results = comparator.compare_with_url(
            topic=topic,
            reference_url=reference_url,
            generation_params={
                'runs': 1,
                'model': 'gpt-4o-mini',
                'temperature': 0.3
            }
        )
        
        # Display results
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        scores = results['scores']
        print(f"\n📊 Similarity Scores:")
        print(f"  • Composite Score:    {scores['composite_score']:.2%}")
        print(f"  • Content Similarity: {scores['content_similarity']:.2%}")
        print(f"  • Structural:        {scores['structural_similarity']:.2%}")
        print(f"  • Code Examples:      {scores['code_similarity']:.2%}")
        print(f"  • Semantic:          {scores['semantic_similarity']:.2%}")
        
        details = results['details']
        print(f"\n📋 Document Details:")
        print(f"  • Reference Length:   {details['reference_length']:,} chars")
        print(f"  • Generated Length:   {details['generated_length']:,} chars")
        print(f"  • Length Ratio:       {details['length_ratio']:.2f}")
        print(f"  • Reference Sections: {len(details.get('common_sections', [])) + len(details.get('missing_sections', []))}")
        print(f"  • Generated Sections: {len(details.get('common_sections', [])) + len(details.get('extra_sections', []))}")
        
        if details.get('missing_sections'):
            print(f"\n  ⚠️  Missing Sections: {', '.join(details['missing_sections'][:3])}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
        
        # Generate full report
        report_path = "comparison_report.md"
        report = comparator.generate_report(results, report_path)
        print(f"\n📄 Full report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo_metrics_only():
    """Demonstrate similarity metrics calculation."""
    
    print("\n" + "="*60)
    print("SIMILARITY METRICS DEMO")
    print("="*60)
    
    text1 = """
    Python list comprehensions provide a concise way to create lists.
    Common applications are to make new lists where each element is 
    the result of some operations applied to each member of another 
    sequence or iterable.
    """
    
    text2 = """
    List comprehensions in Python offer a compact method for list creation.
    They are frequently used to generate new lists by applying operations
    to elements from existing sequences or iterables.
    """
    
    metrics = SimilarityMetrics()
    
    print("\nSample Text 1 (Reference):")
    print(text1.strip())
    
    print("\nSample Text 2 (Generated):")
    print(text2.strip())
    
    print("\n📏 Similarity Metrics:")
    print(f"  • Sequence Similarity: {metrics.sequence_similarity(text1, text2):.2%}")
    print(f"  • Jaccard (words):     {metrics.jaccard_similarity(text1, text2, use_words=True):.2%}")
    print(f"  • Jaccard (chars):     {metrics.jaccard_similarity(text1, text2, use_words=False):.2%}")
    print(f"  • Cosine Similarity:   {metrics.cosine_similarity(text1, text2):.2%}")
    print(f"  • Semantic Similarity: {metrics.semantic_similarity(text1, text2):.2%}")


def demo_downloader():
    """Demonstrate documentation downloader."""
    
    print("\n" + "="*60)
    print("DOCUMENTATION DOWNLOADER DEMO")
    print("="*60)
    
    downloader = DocumentationDownloader()
    
    # Example URLs from different documentation platforms
    urls = {
        "Read the Docs": "https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html",
        "MkDocs Material": "https://squidfunk.github.io/mkdocs-material/getting-started/",
    }
    
    for platform, url in urls.items():
        print(f"\n📥 Downloading from {platform}...")
        print(f"   URL: {url}")
        
        try:
            # Download and extract
            content = downloader.download_and_extract(url)
            
            print(f"\n   ✅ Successfully extracted:")
            print(f"      • Platform: {content['platform']}")
            print(f"      • Sections: {len(content['sections'])}")
            print(f"      • Code Examples: {len(content['code_examples'])}")
            print(f"      • Text Length: {len(content['raw_text'])} chars")
            
            if content['metadata']:
                print(f"      • Title: {content['metadata'].get('title', 'N/A')}")
            
            if content['sections'][:3]:
                print(f"\n   📑 First few sections:")
                for section in content['sections'][:3]:
                    print(f"      - {section['title']}")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Documentation Comparison Demo")
    parser.add_argument('--demo', choices=['comparison', 'metrics', 'downloader', 'all'],
                       default='all', help='Which demo to run')
    
    args = parser.parse_args()
    
    if args.demo == 'comparison' or args.demo == 'all':
        demo_basic_comparison()
    
    if args.demo == 'metrics' or args.demo == 'all':
        demo_metrics_only()
    
    if args.demo == 'downloader' or args.demo == 'all':
        demo_downloader()
    
    print("\n✨ Demo complete!")